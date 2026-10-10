"""Pipeline errors that carry a plain-language message and a suggested next step."""

from __future__ import annotations


class PipelineError(Exception):
    """Base class. ``user_message`` and ``suggestion`` are safe to show to anyone."""

    user_message = "Something went wrong while analysing this star."
    suggestion = "Try again, or pick one of the example stars."

    def __init__(self, detail: str = "", user_message: str | None = None,
                 suggestion: str | None = None):
        super().__init__(detail or self.user_message)
        if user_message:
            self.user_message = user_message
        if suggestion:
            self.suggestion = suggestion


class TargetNotFoundError(PipelineError):
    user_message = "We couldn't find a star with that name or TIC ID."
    suggestion = "Check the spelling, try the TIC number (e.g. 'TIC 420814525'), or pick an example."


class NoDataError(PipelineError):
    user_message = "TESS has not observed this star with a usable light curve."
    suggestion = "Try a different star, for example one from the examples list."


class DataDownloadError(PipelineError):
    user_message = "The NASA data archive did not respond in time."
    suggestion = "This is usually temporary. Wait a minute and try again."


class AnalysisError(PipelineError):
    user_message = "The data was downloaded but could not be analysed."
    suggestion = "The light curve may be too short or too gappy. Try another star."
