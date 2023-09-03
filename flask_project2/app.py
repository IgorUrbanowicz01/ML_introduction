from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
from vectorizer import vect

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
clf = 