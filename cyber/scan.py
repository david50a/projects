from bs4 import BeautifulSoup
import requests
import sys
from urllib.parse import urljoin

s = requests.Session()
s.headers[
    'user-Agent'] = "Your user agent is Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, " \
                    "like Gecko) Chrome/58.0.3029.110 Safari/537.3 "


def get_forms(url):
    soup = BeautifulSoup(s.get(url).content, "html.parser")
    return soup.find_all("form")


def form_details(form):
    detailsOfform = {}
    action = form.attrs.get("action")
    method = form.attrs.get("method", "get")
    inputs = []
    for input_tag in form.find_all("input"):
        input_type = input_tag.get("type", "text")
        input_name = input_tag.get("name")
        input_value = input_tag.attrs.get("value", "")
        inputs.append({"type": input_type, "name": input_name, "value": input_value})
        detailsOfform['action'] = action
        detailsOfform['method'] = method
        detailsOfform["inputs"] = inputs
        return detailsOfform


def vulnerable(response):
    errors = {
        'qouted string not properly terminated,\n unclosed quotation mark after the charachter string,\n you have an '
        'error in you SQL syntax'}
    for error in errors:
        if error in response.content.decode().lower():
            return True
    return False


def sql_injection_scan(url):
    forms = get_forms(url)
    print(f"[+] Detected {len(forms)} forms on {url}.")
    for form in forms:
        details = form_details(form)
        for i in "\"'":
            data = {}
            for input_tag in details["inputs"]:
                if (input_tag['type'] == "hidden" or input_tag["value"]):
                    data[input_tag['name']] = input_tag["value"] + i
                elif input_tag["type"] == "sumbit":
                    data[input_tag['name']] = f"test{i}"
            print(url)
            form_details(form)
        if (details["method"] == "post"):
            res = s.post(url, data=data)
        elif details["method"] == "get":
            res = s.get(url, params=data)
        if vulnerable(res):
            print('SQL Injection attack vulnerablility in ;ink: ', url)
        else:
            print('No SQL Injection attack vulnerablility detected')
            break


if __name__ == "__main__":
    url = "https://cnn.com"
    sql_injection_scan(url)
