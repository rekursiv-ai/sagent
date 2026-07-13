from .client import WebClient
from .slack_response import SlackResponse

"""The Slack Web API allows you to build applications that interact with Slack
in more complex ways than the integrations we provide out of the box."""
__all__ = ["SlackResponse", "WebClient"]
