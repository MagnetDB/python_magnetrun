#!/usr/bin/env python3
"""
Script to query the proposals-for-ct API endpoint
"""

import requests
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://147.173.81.141/api/proposals-for-ct"
TOKEN = "MY_TOKEN"  # Replace with your actual token

def get_proposals(limit=10):
    """
    Fetch proposals from the API (JSON format)
    
    Args:
        limit (int): Maximum number of proposals to retrieve
    
    Returns:
        dict: JSON response from the API
    """
    headers = {
        "Authorization": f"Bearer {TOKEN}"
    }
    
    params = {
        "limit": limit,
        "token": TOKEN
    }
    
    try:
        response = requests.get(API_BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None


def export_csv(output_file="proposals.csv"):
    """
    Export proposals to CSV file
    
    Args:
        output_file (str): Path to save the CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    url = f"{API_BASE_URL}/export-csv"
    
    params = {
        "token": TOKEN
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Save the CSV content to file
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"CSV exported successfully to: {output_file}")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error exporting CSV: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        # Export to CSV
        output_file = sys.argv[2] if len(sys.argv) > 2 else "proposals.csv"
        export_csv(output_file)
    else:
        # Fetch proposals (JSON)
        limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        data = get_proposals(limit=limit)
        
        if data:
            print("Response:")
            print(json.dumps(data, indent=2))
