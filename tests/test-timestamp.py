# --- Example Usage ---
# Filename: M9_Archive_230718-1506.tdms

from python_magnetrun.utils.convert import convert_to_timestamp_aware

date = "230718"
time = "1506"
format_date = "%y%m%d"
format_time = "%H%M"

utc_timestamp, utc_string = convert_to_timestamp_aware(
    date, time, format_date, format_time
)

print(f"Filename Local Time: {date}-{time}")
print(f"UTC Timestamp: {utc_timestamp}")
print(f"UTC Datetime String: {utc_string}")
