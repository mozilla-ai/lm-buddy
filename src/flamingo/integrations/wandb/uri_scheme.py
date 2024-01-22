from enum import Enum


class ArtifactURIScheme(str, Enum):
    """Enumeration of URI schemes to use in a reference artifact."""

    FILE = "file"
    HTTP = "http"
    HTTPS = "https"
    S3 = "s3"
    GCS = "gs"
