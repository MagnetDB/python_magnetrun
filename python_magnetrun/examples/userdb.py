#!/usr/bin/env python3
"""
Script to query the proposals-for-ct API endpoint
"""

import os
import requests
import json

# Configuration
API_BASE_URL = os.getenv("USERDB_SERVER") or "147.173.81.141"
TOKEN = os.getenv("USERDB_API_KEY")


def get_proposals(api_server, headers, limit=10, debug: bool = False):

    url = api_server
    params = {"limit": limit}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None


def export_csv(api_server, headers, output_file="proposals.csv", debug: bool = False):

    url = f"{api_server}/export-csv"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Save the CSV content to file
        with open(output_file, "wb") as f:
            f.write(response.content)

        print(f"CSV exported successfully to: {output_file}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error exporting CSV: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server", help="specify userdb host", type=str, default=API_BASE_URL
    )
    parser.add_argument("--token", help="specify userdb token", type=str, default=TOKEN)
    parser.add_argument(
        "--command", help="specify API command", type=str, default="get-proposals"
    )
    parser.add_argument(
        "--output", help="specify output file", type=str, default="proposals.csv"
    )
    parser.add_argument("--limit", help="number of proposals to fetch", default=10)
    parser.add_argument("--debug", help="enable debug mode", action="store_true")
    args = parser.parse_args()

    url = f"http://{API_BASE_URL}/api/proposals-for-ct"
    if args.server:
        url = f"http://{args.server}/api/proposals-for-ct"
    print("url:", url)

    if args.token:
        TOKEN = args.token

    headers = {"Authorization": f"Bearer {TOKEN}"}
    # headers = {"Authorization": TOKEN}

    if args.command == "export":
        # Export to CSV
        output_file = args.output
        export_csv(url, headers, output_file, debug=args.debug)
    else:
        # Fetch proposals (JSON)
        limit = int(args.limit)
        data = get_proposals(url, headers, limit=limit, debug=args.debug)
        if data:
            print("Response:")
            print(json.dumps(data, indent=2))
