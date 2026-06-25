class ZYStarSolverError(Exception):
    code = "SOLVER_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ValidationError(ZYStarSolverError):
    code = "INVALID_INPUT"


class ModelLoadError(ZYStarSolverError):
    code = "MODEL_LOAD_FAILED"
