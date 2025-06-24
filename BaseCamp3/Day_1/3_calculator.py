# calculator.py
# A simple calculator with add and subtract functions
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field


# Create FastAPI instance with metadata for documentation
app = FastAPI(
    title="Calculator API",
    description="A simple calculator API with add and subtract operations",
    version="0.2.0"
)

# Define Pydantic models for request validation
class CalculationInput(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "a": 10.5,
                    "b": 5.2
                }
            ]
        }
    }


@app.post("/add")
def add(calculation: CalculationInput):
    """Add two numbers and return the result."""
    result = calculation.a + calculation.b
    return {"operation": "add", "a": calculation.a, "b": calculation.b, "result": result}


@app.post("/subtract")
def subtract(calculation: CalculationInput):
    """Subtract b from a and return the result."""
    result = calculation.a - calculation.b
    return {"operation": "add", "a": calculation.a, "b": calculation.b, "result": result}


@app.get("/")
def read_root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Calculator API is running. Use /add or /subtract endpoints."}


# Main program
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9321)