##############################################################################
# Copyright 2018 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
import functools
import inspect
import sys

import msgpack
import rapidjson
from ruamel import yaml

if sys.version_info < (3, 7):
    from rpcq.external.dataclasses import astuple, replace, fields
else:
    from dataclasses import astuple, replace, fields


MAX_BIN_LEN = MAX_STR_LEN = 2 ** 31 - 1  # less than 2 GB
REPR_LIST_TRUNCATION = 10
"Number of list elements to print when calling repr on a Message with a list field."


def repr_value(value):
    """
    Represent a value in human readable form. For long list's this truncates the printed
    representation.

    :param value: The value to represent.
    :return: A string representation.
    :rtype: basestring
    """
    if isinstance(value, list) and len(value) > REPR_LIST_TRUNCATION:
        return "[{},...]".format(", ".join(map(repr, value[:REPR_LIST_TRUNCATION])))
    else:
        return repr(value)


class UnknownMessageType(Exception):
    """Raised when trying to decode an unknown message type."""


class Message:
    """
    Base class for messages.
    """

    def asdict(self):
        """
        Create a dictionary ``{fieldname1: fieldvalue1, ...}`` of the Message object.

        :return: A dictionary representation of the message.
        :rtype: Dict[str,Any]
        """
        return self.__dict__.copy()

    def astuple(self):
        """
        Create a tuple ``{fieldvalue1, ...}`` of the Message object.
         :return: A tuple representation of the message.
        :rtype: Tuple[Any]
        """
        return tuple(getattr(self, f.name) for f in fields(self))

    def replace(self, **kwargs):
        """
        Return a copy of the message object where the fields given in kwargs are
        replaced.

        :param kwargs: The replaced fields.
        :return: A copy of self.
        """
        return replace(self, **kwargs)

    def _extend_by_deprecated_fields(self, d):
        pass

    copy = replace

    def items(self):
        return self.__dict__.items()

    def get(self, key, default):
        return self.__dict__.get(key, default)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join("{}={}".format(k, repr_value(v))
                      for k, v in sorted(self.asdict().items(), key=lambda kv: kv[0])))

    def __eq__(self, other):
        return type(self) == type(other) and astuple(self) == astuple(other)

    def __hash__(self):
        return hash((self.__class__, astuple(self)))

    @staticmethod
    @functools.lru_cache()
    def types():
        """
        Return a mapping ``{type_name: (message_type, args)}`` for all defined Message's,
        where ``args`` is a list of kwarg names that message_type's __init__ function accepts.

        :return: A dictionary of ``Message`` types.
        :rtype: Dict[str,type]
        """
        types = {}
        classes_to_process = [Message]
        while classes_to_process:
            atom = classes_to_process.pop()
            classes_to_process += atom.__subclasses__()
            types[atom.__name__] = (atom, inspect.getfullargspec(atom.__init__).args)

        return types


def _default(obj):
    if isinstance(obj, Message):
        d = obj.__dict__
        obj._extend_by_deprecated_fields(d)
        d["_type"] = obj.__class__.__name__
        return d
    else:
        raise TypeError('Object of type {} is not JSON serializable'.format(obj.__class__.__name__))


def _object_hook(obj):
    if '_type' in obj:
        try:
            class_dict = Message.types()
            msg_type, allowed_args = class_dict[obj['_type']]
        except KeyError:  # pragma no coverage
            raise UnknownMessageType("The message type {} is unknown".format(obj["_type"]))

        itms = {k: v for k, v in obj.items() if k in allowed_args}
        return msg_type(**itms)
    else:
        return obj


def to_msgpack(obj):
    """
    Convert Python objects (including rpcq objects) to a msgpack byte array
    :rtype: bytes
    """
    # Docs for `use_bin_type` parameter are somewhat hard to find so they're copied here:
    #   Use bin type introduced in msgpack spec 2.0 for bytes.
    #   It also enables str8 type for unicode.
    return msgpack.dumps(obj, default=_default, use_bin_type=True)


def from_msgpack(b, *, max_bin_len=MAX_BIN_LEN, max_str_len=MAX_STR_LEN):
    """
    Convert a msgpack byte array into Python objects (including rpcq objects)
    """
    # Docs for raw parameter are somewhat hard to find so they're copied here:
    #   If true, unpack msgpack raw to Python bytes (default).
    #   Otherwise, unpack to Python str (or unicode on Python 2) by decoding with UTF-8 encoding (recommended).
    #   In msgpack >= 0.6, max_xxx_len is reduced from 2 GB to 1 MB, so we set the relevant ones
    #       to 2 GB as to not run into issues with the size of the values returned from rpcq
    return msgpack.loads(b, object_hook=_object_hook, raw=False,
                         max_bin_len=max_bin_len, max_str_len=max_str_len)


def to_json(obj):
    """
    Convert Python objects (including rpcq objects) to a JSON string.
    :rtype: str
    """
    return rapidjson.dumps(obj, default=_default)


def from_json(s):
    """
    Convert a JSON string into Python objects (including rpcq objects).
    """
    return rapidjson.loads(s, object_hook=_object_hook)


def to_yaml_file(obj, f):
    """
    Convert Python objects (including rpcq messages) to yaml and write it to `f`.
    """
    yaml.dump(rapidjson.loads(to_json(obj)), f)


def from_yaml_file(f):
    """
    Read a yaml file and convert to Python objects (including rpcq messages).
    """
    return from_json(to_json(yaml.load(f, Loader=yaml.Loader)))
