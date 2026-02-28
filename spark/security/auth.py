from __future__ import annotations

import getpass
import os


class AdminAuthenticator:
    def authenticate(self) -> bool:
        expected = os.getenv("SPARK_ADMIN_PIN", "1234")
        entered = getpass.getpass("Enter Spark admin PIN: ")
        return entered == expected
