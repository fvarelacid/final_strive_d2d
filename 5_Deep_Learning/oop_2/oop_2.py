from _typeshed import Self

#importing classes from other folders
from folder.file import class


class Person():

    name = "A"
    def __init__(self, surname): 
        self.surname = surname
    

class Employee(Person):
    name = "B"
    money = 1000
    def __init__(self, surname):
       Person.__init__(self, surname)

    def get_name(self):
        return self.name
    
    def set_name(self, name):
        self.name = name

    def die(self):
        self.__empty_bank()
        name = self.get_name()
        name = "shite"

    def __empty_bank(self):
        self.money = 0

bob = Employee("Smith")
 
bob.name += "AAAAAAAA"