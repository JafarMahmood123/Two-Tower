import datetime
from datetime import datetime
from typing import Dict


class UserFeatures:
    def __init__(self, user_id: str, first_name: str, last_name: str,
                 birthdate: str, email: str, password: str,
                 Country: str, City: str):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.birthdate = datetime.strptime(birthdate, '%Y-%m-%d')
        self.age = (datetime.now() - self.birthdate).days // 365
        self.email = email
        self.password = password
        self.country = Country  # New attribute
        self.city = City  # New attribute

    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'age': self.age,
            'first_name': self.first_name,
            'last_name': self.last_name
        }