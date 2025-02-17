import datetime

def convert_to_timestamp(date_str: str, time_str: str, date_format: str="%y%m%d", time_format: str="%H%M") -> tuple:
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

    # Format the datetime object to the desired string format
    formatted_date_time = date_time_obj.strftime("%Y-%m-%d %H:%M:%S")

    return (timestamp, formatted_date_time)
