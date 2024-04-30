from pydantic import BaseModel

class Vehicle(BaseModel):
    _id: str
    make: str
    vehicleModel: str
    manufacturedYear: int
    registeredYear: int
    mileage: int
    numberOfPreviousOwners: int
    exteriorColor: str
    fuelType: str
    condition: str
    transmission: str
    bodyType: str
    engineCapacity: int
    value: int