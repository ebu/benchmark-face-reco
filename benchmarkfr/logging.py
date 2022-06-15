import json
import logging.config


class StructuredMessage:
    def __init__(self, message, **kwargs):
        self.message = message
        self.kwargs = kwargs

    def __str__(self):
        return "%s %s" % (self.message, json.dumps(self.kwargs))


_ = StructuredMessage


def configure(log_level):
    logging.config.dictConfig({"version": 1, "disable_existing_loggers": True,
                               "handlers": {"console": {"level": log_level, "class": "logging.StreamHandler",
                                                        "stream": "ext://sys.stderr"}},
                               "root": {"handlers": ["console"], "level": log_level},
                               "loggers": {"benchmarkfr": {}, "__main__": {}}})
