def individual_serializer(vehicle) -> dict:
    return {
        '_id': str(vehicle["_id"]),
        'make': vehicle['make'].lower(),
        'vehicleModel': vehicle['vehicleModel'].lower(),
        'manufacturedYear': vehicle['manufacturedYear'],
        'registeredYear': vehicle['registeredYear'],
        'mileage': vehicle['mileage'],
        'previousOwners': vehicle['numberOfPreviousOwners'],
        'exteriorColor': vehicle['exteriorColor'].lower(),
        'fuelType': vehicle['fuelType'].lower(),
        'condition': vehicle['condition'].lower(),
        'transmission': vehicle['transmission'].lower(),
        'bodyType': vehicle['bodyType'].lower(),
        'engineCapacity': vehicle['engineCapacity'],
        'value': vehicle['currentPrice']
    }

    
def list_serializer(vehicles) -> list:
    return [individual_serializer(vehicle) for vehicle in vehicles]    
    