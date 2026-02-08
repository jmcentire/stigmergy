"""Static file providers — CSV team roster, email aliases, Linear markdown.

These extract the current behavior from graph/identity.py into the
IdentityProvider protocol. They are the authoritative source: live
providers enrich but do not override static data.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from stigmergy.identity.provider import PersonRecord

logger = logging.getLogger(__name__)


class StaticCSVProvider:
    """Load person records from team roster CSV and optional email aliases CSV.

    CSV format (no header row):
        Name,Name <email>,@Slack Handle,github_handle

    Email aliases CSV format (# comments allowed):
        work_email,alias_email
    """

    def __init__(
        self,
        team_csv: Path,
        alias_csv: Path | None = None,
    ) -> None:
        self._team_csv = team_csv
        self._alias_csv = alias_csv

    def source_name(self) -> str:
        return "static-csv"

    def available(self) -> bool:
        return self._team_csv.exists()

    def fetch(self) -> list[PersonRecord]:
        if not self._team_csv.exists():
            return []

        # First pass: load team roster
        records: list[PersonRecord] = []
        email_to_record: dict[str, PersonRecord] = {}

        try:
            with open(self._team_csv) as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 4:
                        continue

                    name = row[0].strip()
                    email_field = row[1].strip()
                    slack_raw = row[2].strip()
                    github = row[3].strip()

                    # Parse email from "Name <email>" format
                    email_match = re.search(r"<([^>]+)>", email_field)
                    email = email_match.group(1) if email_match else email_field

                    slack = slack_raw.lstrip("@").strip()

                    record = PersonRecord(
                        name=name,
                        email=email.lower(),
                        github_handle=github,
                        slack_handle=slack,
                        source="static-csv",
                    )
                    records.append(record)
                    email_to_record[email.lower()] = record
        except OSError as exc:
            logger.warning("Failed to read team roster %s: %s", self._team_csv, exc)
            return []

        logger.info("Loaded %d team members from %s", len(records), self._team_csv)

        # Second pass: load email aliases
        if self._alias_csv and self._alias_csv.exists():
            alias_count = 0
            try:
                with open(self._alias_csv) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row or row[0].strip().startswith("#"):
                            continue
                        if len(row) < 2:
                            continue

                        work_email = row[0].strip().lower()
                        alias_email = row[1].strip().lower()

                        record = email_to_record.get(work_email)
                        if record:
                            record.aliases.append(alias_email)
                            # Also add username part (skip noreply-style)
                            alias_user = alias_email.split("@")[0]
                            if "+" not in alias_user:
                                record.aliases.append(alias_user)
                            alias_count += 1
            except OSError as exc:
                logger.warning("Failed to read aliases %s: %s", self._alias_csv, exc)

            logger.info("Loaded %d email aliases from %s", alias_count, self._alias_csv)

        return records


class StaticLinearMdProvider:
    """Load Linear UUIDs and team assignments from linear-reference.md.

    Parses the markdown table under "## Team Members" with format:
        | Name | Email | UUID |
    """

    def __init__(self, md_path: Path) -> None:
        self._md_path = md_path

    def source_name(self) -> str:
        return "static-linear-md"

    def available(self) -> bool:
        return self._md_path.exists()

    def fetch(self) -> list[PersonRecord]:
        if not self._md_path.exists():
            return []

        try:
            text = self._md_path.read_text()
        except OSError as exc:
            logger.warning("Failed to read %s: %s", self._md_path, exc)
            return []

        member_pattern = re.compile(
            r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|"
        )

        records: list[PersonRecord] = []
        in_members = False

        for line in text.splitlines():
            if "Team Members" in line:
                in_members = True
                continue
            if line.startswith("###") or (line.startswith("## ") and in_members):
                in_members = False
                continue
            if not in_members:
                continue

            match = member_pattern.match(line)
            if not match:
                continue

            name = match.group(1).strip()
            email = match.group(2).strip().lower()
            linear_uuid = match.group(3).strip()

            # Skip header row
            if name.lower() == "name" or email.lower() == "email":
                continue

            records.append(
                PersonRecord(
                    name=name,
                    email=email,
                    linear_uuid=linear_uuid,
                    source="static-linear-md",
                )
            )

        logger.info("Loaded %d Linear members from %s", len(records), self._md_path)
        return records
