from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import os, csv


# Create FastAPI instance with metadata for documentation
app = FastAPI(
    title="Products API",
    description="A simple Products API with CRUD operations",
    version="0.2.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_FILE = "6a_products.csv"

# Define the data model
class Record(BaseModel):
    product_id: int
    product_name: str
    product_description: str
    product_price: float
    product_in_stock_qty: int


@app.get("/")
def read_root():
    return {"message": "Simple CSV Record API"}

@app.post("/records/")
def create_record(record: Record):
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([record.product_id, record.product_name, record.product_description,record.product_price,record.product_in_stock_qty])

    return {"status": "success", "message": "Record added successfully"}


@app.get("/records/{record_id}")
def read_record_by_path(record_id: int):
    with open(CSV_FILE, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if "product_id" in row and row["product_id"].isdigit() and int(row["product_id"]) == record_id:
                return row

@app.get("/record")
def read_record_by_query(record_id: int):
    with open(CSV_FILE, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if "product_id" in row and row["product_id"].isdigit() and int(row["product_id"]) == record_id:
                return row



@app.delete("/record")
def delete_record(id: int, api_key: str):
    # Read all records
    api_keys=['abc123','dev456']
    records = []
    if api_key not in api_keys:
      return {"status": "Denied", "message": "you don't have permission to delete record"}
    record_found = False

    with open(CSV_FILE, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
          if int(row["product_id"]) == id:
              record_found = True
              continue  # Skip this record (delete it)
          records.append(row)

    with open(CSV_FILE, mode='w', newline='') as file:
        fieldnames = ["product_id", "product_name", "product_description","product_price","product_in_stock_qty"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return {"status": "success", "message": "Record deleted successfully"}

@app.get("/")
def read_root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Calculator API is running. Use /add or /subtract endpoints."}


# Main program
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9321)