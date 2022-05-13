#!/usr/bin/env python
#
# Copyright (c) 2009, 2010, Henry Precheur <henry@precheur.org>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
#
'''Formats dates according to the :RFC:`3339`.

Report bugs and feature requests on Sourcehut_

Source availabe on this Mercurial repository: https://hg.sr.ht/~henryprecheur/rfc3339

.. _Sourcehut: https://todo.sr.ht/~henryprecheur/rfc3339
'''

__author__ = 'Henry Precheur <henry@precheur.org>'
__license__ = 'ISCL'
__version__ = '6.2'
__all__ = ('rfc3339', )

from datetime import (
    datetime,
    date,
    timedelta,
    tzinfo,
)
import time
import unittest

def _timezone(utc_offset):
    '''
    Return a string representing the timezone offset.

    >>> _timezone(0)
    '+00:00'
    >>> _timezone(3600)
    '+01:00'
    >>> _timezone(-28800)
    '-08:00'
    >>> _timezone(-8 * 60 * 60)
    '-08:00'
    >>> _timezone(-30 * 60)
    '-00:30'
    '''
    # Python's division uses floor(), not round() like in other languages:
    #   -1 / 2 == -1 and not -1 / 2 == 0
    # That's why we use abs(utc_offset).
    hours = abs(utc_offset) // 3600
    minutes = abs(utc_offset) % 3600 // 60
    sign = (utc_offset < 0 and '-') or '+'
    return '%c%02d:%02d' % (sign, hours, minutes)

def _timedelta_to_seconds(td):
    '''
    >>> _timedelta_to_seconds(timedelta(hours=3))
    10800
    >>> _timedelta_to_seconds(timedelta(hours=3, minutes=15))
    11700
    >>> _timedelta_to_seconds(timedelta(hours=-8))
    -28800
    '''
    return int((td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6)

def _utc_offset(timestamp, use_system_timezone):
    '''
    Return the UTC offset of `timestamp`. If `timestamp` does not have any `tzinfo`, use
    the timezone informations stored locally on the system.

    >>> if time.localtime().tm_isdst:
    ...     system_timezone = -time.altzone
    ... else:
    ...     system_timezone = -time.timezone
    >>> _utc_offset(datetime.now(), True) == system_timezone
    True
    >>> _utc_offset(datetime.now(), False)
    0
    '''
    if (isinstance(timestamp, datetime) and
            timestamp.tzinfo is not None):
        return _timedelta_to_seconds(timestamp.utcoffset())
    elif use_system_timezone:
        if timestamp.year < 1970:
            # We use 1972 because 1970 doesn't have a leap day (feb 29)
            t = time.mktime(timestamp.replace(year=1972).timetuple())
        else:
            t = time.mktime(timestamp.timetuple())
        if time.localtime(t).tm_isdst: # pragma: no cover
            return -time.altzone
        else:
            return -time.timezone
    else:
        return 0

def _string(d, timezone):
    return ('%04d-%02d-%02dT%02d:%02d:%02d%s' %
            (d.year, d.month, d.day, d.hour, d.minute, d.second, timezone))

def _string_milliseconds(d, timezone):
    return ('%04d-%02d-%02dT%02d:%02d:%02d.%03d%s' %
            (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond / 1000, timezone))

def _string_microseconds(d, timezone):
    return ('%04d-%02d-%02dT%02d:%02d:%02d.%06d%s' %
            (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond, timezone))

def _format(timestamp, string_format, utc, use_system_timezone):
    # Try to convert timestamp to datetime
    try:
        if use_system_timezone:
            timestamp = datetime.fromtimestamp(timestamp)
        else:
            timestamp = datetime.utcfromtimestamp(timestamp)
    except TypeError:
        pass

    if not isinstance(timestamp, date):
        raise TypeError('Expected timestamp or date object. Got %r.' %
                        type(timestamp))

    if not isinstance(timestamp, datetime):
        timestamp = datetime(*timestamp.timetuple()[:3])
    utc_offset = _utc_offset(timestamp, use_system_timezone)
    if utc:
        # local time -> utc
        return string_format(timestamp - timedelta(seconds=utc_offset), 'Z')
    else:
        return string_format(timestamp , _timezone(utc_offset))

def format_millisecond(timestamp, utc=False, use_system_timezone=True):
    '''
    Same as `rfc3339.format` but with the millisecond fraction after the seconds.
    '''
    return _format(timestamp, _string_milliseconds, utc, use_system_timezone)

def format_microsecond(timestamp, utc=False, use_system_timezone=True):
    '''
    Same as `rfc3339.format` but with the microsecond fraction after the seconds.
    '''
    return _format(timestamp, _string_microseconds, utc, use_system_timezone)

def format(timestamp, utc=False, use_system_timezone=True):
    '''
    Return a string formatted according to the :RFC:`3339`. If called with
    `utc=True`, it normalizes `timestamp` to the UTC date. If `timestamp` does
    not have any timezone information, uses the local timezone::

        >>> d = datetime(2008, 4, 2, 20)
        >>> rfc3339(d, utc=True, use_system_timezone=False)
        '2008-04-02T20:00:00Z'
        >>> rfc3339(d) # doctest: +ELLIPSIS
        '2008-04-02T20:00:00...'

    If called with `use_system_timezone=False` don't use the local timezone if
    `timestamp` does not have timezone informations and consider the offset to UTC
    to be zero::

        >>> rfc3339(d, use_system_timezone=False)
        '2008-04-02T20:00:00+00:00'

    `timestamp` must be a `datetime`, `date` or a timestamp as
    returned by `time.time()`::

        >>> rfc3339(0, utc=True, use_system_timezone=False)
        '1970-01-01T00:00:00Z'
        >>> rfc3339(date(2008, 9, 6), utc=True,
        ...         use_system_timezone=False)
        '2008-09-06T00:00:00Z'
        >>> rfc3339(date(2008, 9, 6),
        ...         use_system_timezone=False)
        '2008-09-06T00:00:00+00:00'
        >>> rfc3339('foo bar') # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: Expected timestamp or date object. Got <... 'str'>.

    For dates before January 1st 1970, the timezones will be the ones used in
    1970. It might not be accurate, but on most sytem there is no timezone
    information before 1970.
    '''
    return _format(timestamp, _string, utc, use_system_timezone)

# FIXME deprecated
rfc3339 = format

class LocalTimeTestCase(unittest.TestCase):
    '''
    Test the use of the timezone saved locally. Since it is hard to test using
    doctest.
    '''

    def setUp(self):
        local_utcoffset = _utc_offset(datetime.now(),
                                      use_system_timezone=True)
        self.local_utcoffset = timedelta(seconds=local_utcoffset)
        self.local_timezone = _timezone(local_utcoffset)

    def test_datetime(self):
        d = datetime.now()
        self.assertEqual(rfc3339(d),
                         d.strftime('%Y-%m-%dT%H:%M:%S') + self.local_timezone)

    def test_datetime_timezone(self):

        class FixedNoDst(tzinfo):
            'A timezone info with fixed offset, not DST'

            def utcoffset(self, dt):
                return timedelta(hours=2, minutes=30)

            def dst(self, dt):
                return None

        fixed_no_dst = FixedNoDst()

        class Fixed(FixedNoDst):
            'A timezone info with DST'
            def utcoffset(self, dt):
                return timedelta(hours=3, minutes=15)

            def dst(self, dt):
                return timedelta(hours=3, minutes=15)

        fixed = Fixed()

        d = datetime.now().replace(tzinfo=fixed_no_dst)
        timezone = _timezone(_timedelta_to_seconds(fixed_no_dst.\
                                                   utcoffset(None)))
        self.assertEqual(rfc3339(d),
                         d.strftime('%Y-%m-%dT%H:%M:%S') + timezone)

        d = datetime.now().replace(tzinfo=fixed)
        timezone = _timezone(_timedelta_to_seconds(fixed.dst(None)))
        self.assertEqual(rfc3339(d),
                         d.strftime('%Y-%m-%dT%H:%M:%S') + timezone)

    def test_datetime_utc(self):
        d = datetime.now()
        d_utc = d - self.local_utcoffset
        self.assertEqual(rfc3339(d, utc=True),
                         d_utc.strftime('%Y-%m-%dT%H:%M:%SZ'))

    def test_date(self):
        d = date.today()
        self.assertEqual(rfc3339(d),
                         d.strftime('%Y-%m-%dT%H:%M:%S') + self.local_timezone)

    def test_date_utc(self):
        d = date.today()
        # Convert `date` to `datetime`, since `date` ignores seconds and hours
        # in timedeltas:
        # >>> date(2008, 9, 7) + timedelta(hours=23)
        # date(2008, 9, 7)
        d_utc = datetime(*d.timetuple()[:3]) - self.local_utcoffset
        self.assertEqual(rfc3339(d, utc=True),
                         d_utc.strftime('%Y-%m-%dT%H:%M:%SZ'))

    def test_timestamp(self):
        d = time.time()
        self.assertEqual(
            rfc3339(d),
            datetime.fromtimestamp(d).
            strftime('%Y-%m-%dT%H:%M:%S') + self.local_timezone)

    def test_timestamp_utc(self):
        d = time.time()
        # utc -> local timezone
        d_utc = datetime.utcfromtimestamp(d) + self.local_utcoffset
        self.assertEqual(rfc3339(d),
                         (d_utc.strftime('%Y-%m-%dT%H:%M:%S') +
                          self.local_timezone))

    def test_before_1970(self):
        d = date(1885, 1, 4)
        self.assertTrue(rfc3339(d).startswith('1885-01-04T00:00:00'))
        self.assertEqual(rfc3339(d, utc=True, use_system_timezone=False),
                         '1885-01-04T00:00:00Z')

    def test_1920(self):
        d = date(1920, 2, 29)
        x = rfc3339(d, utc=False, use_system_timezone=True)
        self.assertTrue(x.startswith('1920-02-29T00:00:00'))

    # If these tests start failing it probably means there was a policy change
    # for the Pacific time zone.
    # See http://en.wikipedia.org/wiki/Pacific_Time_Zone.
    if 'PST' in time.tzname:
        def testPDTChange(self):
            '''Test Daylight saving change'''
            # PDT switch happens at 2AM on March 14, 2010

            # 1:59AM PST
            self.assertEqual(rfc3339(datetime(2010, 3, 14, 1, 59)),
                             '2010-03-14T01:59:00-08:00')
            # 3AM PDT
            self.assertEqual(rfc3339(datetime(2010, 3, 14, 3, 0)),
                             '2010-03-14T03:00:00-07:00')

        def testPSTChange(self):
            '''Test Standard time change'''
            # PST switch happens at 2AM on November 6, 2010

            # 0:59AM PDT
            self.assertEqual(rfc3339(datetime(2010, 11, 7, 0, 59)),
                             '2010-11-07T00:59:00-07:00')

            # 1:00AM PST
            # There's no way to have 1:00AM PST without a proper tzinfo
            self.assertEqual(rfc3339(datetime(2010, 11, 7, 1, 0)),
                             '2010-11-07T01:00:00-07:00')

    def test_millisecond(self):
        x = datetime(2018, 9, 20, 13, 11, 21, 123000)
        self.assertEqual(
            format_millisecond(
                datetime(2018, 9, 20, 13, 11, 21, 123000),
                utc=True,
                use_system_timezone=False),
            '2018-09-20T13:11:21.123Z')

    def test_microsecond(self):
        x = datetime(2018, 9, 20, 13, 11, 21, 12345)
        self.assertEqual(
            format_microsecond(
                datetime(2018, 9, 20, 13, 11, 21, 12345),
                utc=True,
                use_system_timezone=False),
            '2018-09-20T13:11:21.012345Z')

if __name__ == '__main__': # pragma: no cover
    import doctest
    doctest.testmod()
    unittest.main()
