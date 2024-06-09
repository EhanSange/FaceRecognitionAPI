from geopy.distance import geodesic

#UIN
#ALLOWED_LOCATION = (-6.930438390912229, 107.71771084088306)
#AING
ALLOWED_LOCATION = (-6.9199573, 107.7207022) 
ALLOWED_RADIUS = 200  # Radius

def is_within_allowed_radius(user_location):
    distance = geodesic(ALLOWED_LOCATION, user_location).meters
    return distance <= ALLOWED_RADIUS
