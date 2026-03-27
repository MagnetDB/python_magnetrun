import logging
from datetime import datetime

import pytz  # You will need to install this library: pip install pytz

logger = logging.getLogger(__name__)


def convert_to_timestamp_aware(
    date_str: str,
    time_str: str,
    date_format: str = "%y%m%d",
    time_format: str = "%H%M",
    time_zone: str = "Europe/Paris",  # Default to the French Time Zone
) -> tuple:
    """
    Converts date and time strings to a UTC timestamp, explicitly handling the
    time zone of the input strings.

    :param date_str: Date string (e.g., '230718')
    :param time_str: Time string (e.g., '1506')
    :param date_format: Format string for the date part (e.g., '%y%m%d')
    :param time_format: Format string for the time part (e.g., '%H%M')
    :param time_zone: The time zone of the input date/time (e.g., 'Europe/Paris')
    :return: A tuple containing the UTC timestamp (float) and a UTC formatted datetime string.
    """

    # 1. Parse the date and time strings into a NAIVE datetime object
    date_time_str = date_str + time_str
    date_time_format = date_format + time_format
    naive_dt = datetime.strptime(date_time_str, date_time_format)

    # 2. Define the input time zone
    tz = pytz.timezone(time_zone)

    # 3. Make the datetime object TIME ZONE AWARE
    # This assigns the Paris time zone to the time 15:06
    aware_dt_local = tz.localize(naive_dt)

    # 4. Convert the TIME ZONE AWARE object to UTC
    aware_dt_utc = aware_dt_local.astimezone(pytz.utc)

    # 5. Convert the UTC datetime object to a timestamp (which is always UTC seconds since the epoch)
    timestamp = aware_dt_utc.timestamp()

    # 6. Format the UTC datetime object to the desired string format
    # The string will represent the UTC time (e.g., 2023-07-18 13:06:00)
    formatted_date_time_utc = aware_dt_utc.strftime("%Y-%m-%dT%H:%M:%S")

    # The formatted string can optionally include 'Z' for UTC or the offset
    # formatted_date_time_utc = aware_dt_utc.strftime("%Y-%m-%dT%H:%M:%S%z")

    return (timestamp, formatted_date_time_utc)


def convert_to_timestamp(
    date_str: str, time_str: str, date_format: str = "%y%m%d", time_format: str = "%H%M"
) -> tuple:
    """_summary_

    ex for tdms files

    date_format = "%y%m%d"
    time_format = "%H%M%S"

    ex for pupitre files

    date_format = "%Y%m%d"
    time_format = "%H:%M:%S"

    :param date_str: _description_
    :type date_str: _type_
    :param time_str: _description_
    :type time_str: _type_
    :return: _description_
    :rtype: _type_
    """

    from datetime import datetime

    # Format the date and time strings

    # Parse the date and time strings into a datetime object
    date_time_str = date_str + time_str
    date_time_format = date_format + time_format
    date_time_obj = datetime.strptime(date_time_str, date_time_format)

    # Convert the datetime object to a timestamp
    timestamp = date_time_obj.timestamp()
    # print(f"convert_to_timestamp: timestamp={timestamp}, type={type(timestamp)}")

    # Format the datetime object to the desired string format
    formatted_date_time = date_time_obj.strftime("%Y-%m-%d %H:%M:%S")

    return (timestamp, formatted_date_time)
