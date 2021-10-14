from flask import Flask
from flask_restful import Api, Resource, reqparse

app1 = Flask(__name__)

api = Api(app1)