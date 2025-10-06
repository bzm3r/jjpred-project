"""Global variables representing path information (where the main project folder
is located, where the excel files are located, etc)."""

from __future__ import annotations

import os
from pathlib import Path
import getpass

ASSUMED_FOLDER_NAME: str = "jjpred-project"
"""Assumed name of the main project folder."""

USER_NAME = getpass.getuser()
"""User's login name."""


def get_main_folder(assumed_folder_name: str) -> Path:
    """Get the full path of the main folder, given its assumed name."""
    cwd = Path(os.getcwd()).absolute()
    # Traverse up from the given path to the root
    for parent in cwd.parents:
        if parent.name == assumed_folder_name:
            return parent

    if cwd.name == assumed_folder_name:
        return cwd

    raise ValueError(
        f"Current working directory does not contain {ASSUMED_FOLDER_NAME}!"
    )


USER_HOME_FOLDER: Path = Path(f"C:\\Users\\{USER_NAME}")

USER_ONEDRIVE_FOLDER: Path = USER_HOME_FOLDER.joinpath(
    "OneDrive - Jan and Jul"
)

BRIAN_TWK_FOLDER: Path = USER_ONEDRIVE_FOLDER.joinpath("TWK-Brian")

MAIN_FOLDER: Path = get_main_folder(ASSUMED_FOLDER_NAME)
"""Full path of the main folder."""

ANALYSIS_INPUT_FOLDER: Path = MAIN_FOLDER.joinpath("AnalysisInput")
"""Full path of the folder where the analysis input data is located."""

ANALYSIS_OUTPUT_FOLDER: Path = MAIN_FOLDER.joinpath("AnalysisOutput")
"""Full path of the folder where analysis output data will be placed."""
