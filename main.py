import streamlit as st

from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from streamlit.server.server_util import config

from tornado.web import RequestHandler, Application


"""
Streamlit sharing offers three free slots for deploying streamlit applications.
For tiny apps it is prohibitive to use a whole slot, instead, it would be much better
to bundle them into one app and relay to the specific application depending on some
url parameters.

Unfortunately, URL params are not accessible right now.  
"""

# NOT WORKING FOR NOW

class Handler(RequestHandler):
    def get(self, **kwargs):
        print(kwargs)
        key = kwargs.get('key')
        # I prefer dict.get() here, since if you change the `+` to a `*`,
        # it's possible that no key was supplied, and kwargs['key']
        # will throw a KeyError exception


st.header('Hello!')

Server.get_current()._ioloop.add_handlers(Application, host_pattern='localhost:8501', host_handlers=[(r"/", Handler),])
