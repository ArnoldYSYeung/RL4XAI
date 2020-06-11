"""
Functions for data collection, processing, analysis, etc.

Date :                  December 7th, 2019

"""

import requests
import json
import pandas as pd
from pandas.io.json import json_normalize

from private_variables import *

def save_json_as_csv(data, save_file):
    """
    Saves data in json format as a csv.
    """
    df = json_normalize(data)
    df.to_csv(save_file)    

def get_json_response(url, auth_key="", save_json="", save_csv=""):
    """
    Runs GET command to retrieve data and convert to json format. Optional
    saving in .json and .csv.
    Arguments:
        url (string) :                  GET command to run
        auth_key (string) :             authorization key
        save_json (string/Boolean) :    file location to save .json file (does not save if "")
        save_csv (string) :             file location to save .csv file (does not save if "")
    """

    headers = {'Authorization': 'token '+auth_key}

    response = requests.get(url, headers=headers)
    
    data = None
    if "results" in response.json():
        data = response.json()["results"]  #   list of dictionaries (each dictionary = 1 entry)
    else:
        data = response.json()

    #   recurse for next pages
    if "next" in response.json():
        next_url = response.json()["next"]
        if next_url is not None:
            #print(next_url)
            next_data, response = get_json_response(next_url, auth_key=auth_key)
            data += next_data

    if save_json != "":
        #   save to .json file
        with open(save_json, 'w+') as file:
            file.write(json.dumps(data))
        print(save_json+" saved.")

    if save_csv != "":
        #   save to .csv file    
        save_json_as_csv(data, save_csv)
        print(save_csv+" saved.")
    
    return data, response

def post_object(url, params={}, json=[], auth_key=""):
    """
    Runs a POST command to send information to server.
    Arguments:
        url (str) :             POST command to run
        params (dict) :         body parameters of the POST command
        json (lst[dict]) :      list of json encoded data
        auth_key (str) :        authorization key to be in header
    Returns:
        response from POST command
        
    See stackoverflow.com/questions/31168819/
    """
    
    headers = {'Authorization': 'token '+str(auth_key)}
    
    response = requests.post(url, data=params, json=json, headers=headers)
    return response
    
def create_environment(url, name, auth_key=""):
    """
    Creates an environment.
    Arguments:
        url (str) :             URL for POST command (e.g. http://mooclet.dgp.toronto.edu/engine/api/v1)
        name (str) :            name of environment
        auth_key (str) :        authorization key
    
    """
    
    if url[-1] != "/":
        url += "/"
    url += "/environment"    
    
    params = {'name': name}
    
    return post_object(url, params=params, auth_key=auth_key)

def get_environment(url, id=None, auth_key=""):
    
    if url[-1] != "/":
        url += "/"
    
    if id is None:
        url += "environment"
    else:
        url += "environment/"+str(id)
        
    return requests.get(url, headers={'Authorization': 'token '+str(auth_key)})

def create_mooclet(url, name, environment, policy, auth_key):
    
    if url[-1] != "/":
        url += "/"
    url += "mooclet"    
    
    print(url)
    
    params = {'name': name, 'environment': environment, 'policy': policy}
    
    return post_object(url, params=params, auth_key=auth_key)

def get_mooclet(url, id=None, auth_key=""):
    
    if url[-1] != "/":
        url += "/"
    
    if id is None:
        url += "mooclet"
    else:
        url += "mooclet/"+str(id)
        
    return requests.get(url, headers={'Authorization': 'token '+str(auth_key)})

def create_learner(url, learner, auth_key=""):
    """
    Create a learner.
    Arguments:
        url (str) :                     URL for POST command
        learner (dict{}) :              learner dictionary containing the POST parameters
            "name" (str) :              name of learner
            "environment" (int) :       environment number (leave None)
            "learner_id" (int) :        id of learner
        auth_key (str) :                authorization key
    """
    if url[-1] != "/":
        url += "/"
    url += "/learner"
    
    return post_object(url, json=learner, auth_key=auth_key)

def create_version(url, version, auth_key=""):
    """
    Create versions.
    Arguments:
        url (str) :                     URL for POST command
        version (dict{}) :              version dictionary containing the POST parameters
            "mooclet" (int) :           mooclet id
            "name" (str) :              name of version
            "text" (str) :              text associated with version
        auth_key (str) :                authorization key
    """

    if url[-1] != "/":
        url += "/"
    url += "/version"
    
    return post_object(url, json=version, auth_key=auth_key)

def get_version(url, id=None, auth_key=""):
    """
    Get a version with a specified id.
    Arguments:
        url (str) :                 URL for database
        id (int) :                  id for version
        auth_key (str) :            authorization key
    """
    
    if url[-1] != "/":
        url += "/"
    
    if id is None:
        url += "version"
    else:
        url += "version/"+str(id)
        
    return requests.get(url, headers={'Authorization': 'token '+str(auth_key)})

def create_list_versions(versions, url, auth_key):
    """
    Create versions within a specified mooclet.
    Arguments:
        versions(list[dict{}, ]) :                  list of version attributes
            'name' (str) :                          name of version
            'text' (str) :                          text corresponding with version
            'mooclet' (str) :                       mooclet id
            'version_json' (dict) :                 (currently does not support)
        url (str) :                                 URL for POST command
        auth_key (str) :                            authorization key
    """
    
    data = []
    for entry in versions:
        name = entry['name']
        text = entry['text']
        mooclet_id = entry['mooclet']
        params = {'name': name, 'text': text, 'mooclet': mooclet_id}
        data.append(create_version(url, params, auth_key=auth_key))
    
    return data

def import_versions_to_new_mooclet(url, source_id, target_id, auth_key=""):
    """
    Imports all existing versions within a source mooclet to a target mooclet.
    Arguments:
        url (str) :                     URL for server
        source_id (int) :               id of source mooclet (not mooclet_id)
        target_id (int) :               id of target mooclet (not mooclet_id)
        auth_key (str) :                authorization key
    """
    
    if url[-1] != "/":
        url += "/"
    version_url = url + "version"
    
    #   get ids of versions we want to copy over
    requirements ={"mooclet": source_id}
    version_ids = get_ids_meet_requirements(version_url, 
                                            requirements=requirements, 
                                            auth_key=auth_key)
    
    #   get versions
    versions = []                   #   list of dictionaries
    for id in version_ids:
        version = get_version(url, id=id, auth_key=auth_key).json()
        version['mooclet'] = target_id
        versions.append(version)
    
    #   add versions to target mooclet
    headers = {'Authorization': 'token '+str(auth_key)}
    data = create_list_versions(versions, url, auth_key)
    
    return data

def create_variable(url, variable, auth_key=""):
    """
    Create variable.
    Arguments:
        url (str) :                     URL for POST command
        variable (dict{}) :             variable dictionary containing the POST parameters
            "name" (str) :              name of variable
            "environment" (int) :       environment id
        auth_key (str) :                authorization key
    """

    if url[-1] != "/":
        url += "/"
    url += "/variable"
    
    return post_object(url, json=variable, auth_key=auth_key)

def get_variable(url, id=None, auth_key=""):
    
    if url[-1] != "/":
        url += "/"
    
    if id is None:
        url += "variable"
    else:
        url += "variable/"+str(id)
        
    return requests.get(url, headers={'Authorization': 'token '+str(auth_key)})

def create_list_variables(variable_names, environment, url, auth_key):
    """
    Create variables within a specified environment.
    Arguments:
        variable_names(list[str, ]) :               list of variable names
        environment (int) :                         environment id
        url (str) :                                 URL for POST command
        auth_key (str) :                            authorization key
    """
    
    variables = []
    data = []
    for name in variable_names:
        params = {'name': name, 'environment': environment}
        data.append(create_variable(url, params, auth_key=auth_key))
    print(variables)
    
    return data

def import_variables_to_new_environment(url, source_id, target_id, auth_key=""):
    """
    Imports all existing versions within a source mooclet to a target mooclet.
    Arguments:
        url (str) :                     URL for server
        source_id (int) :               id of source environment
        target_id (int) :               id of target environment
        auth_key (str) :                authorization key
    """
    
    if url[-1] != "/":
        url += "/"
    variable_url = url + "variable"
    
    #   get ids of versions we want to copy over
    requirements ={"environment": source_id}
    variable_ids = get_ids_meet_requirements(variable_url, 
                                            requirements=requirements, 
                                            auth_key=auth_key)
    
    #   get variables
    variables = []                   #   list of strings
    for id in variable_ids:
        variable = get_variable(url, id=id, auth_key=auth_key).json()
        variables.append(variable["name"])
    
    #   add variables to target environment
    headers = {'Authorization': 'token '+str(auth_key)}
    data = create_list_variables(variables, target_id, url, auth_key)
    
    return data
    
def modify_request(url, id, params, auth_key=""):
    """
    Used for modifying an element in the database.
    Arguments:
        url (str) :                             full URL for PUT command
        id (int) :                              id of object to modify
        params (dict{attr: value, }) :          attributes of object to modify
        auth_key (str) :                        authorization key
    """
    
    header = {'Authorization': 'token '+auth_key}
    
    if url[-1] != "/":
        url += "/"
    
    url = url+str(id)
    print(url)
    
    return requests.put(url, data=params, headers=header)

def get_ids_meet_requirements(url, requirements={}, auth_key=""):
    """
    Get all entry ids which meets the requirements.
    Arguments:
        url (str) :                     URL for GET command
        requirements (dict{,}) :        required parameters for id
        auth_key (str) :                authorization key
    Returns:
        list of ids which meet requirements (list[int, ])
    """
    
    #   Get ID of entries which meet requirements
    ids = []
    data = requests.get(url, headers={'Authorization': 'token '+ str(auth_key)})
        
    if "results" not in data.json():
        print(data)
        raise ValueError("results not found")
    
    for entry in data.json()['results']:
        if not requirements:
            ids.append(entry["id"])
        else:
            meets_required = True
            for require_key in requirements:
                if require_key not in entry or entry[require_key] != requirements[require_key]:
                    meets_required = False
            if meets_required:
                ids.append(entry["id"])
    
    if data.json()['next']:
        url = data.json()['next']
        new_ids = get_ids_meet_requirements(url, requirements, auth_key)
        ids += new_ids
    
    return ids
    
def get_values_meet_requirements(url, requirements={}, variables=[], auth_key="", 
                                 dtypes=["value"]):
    """
    Get values with corresponding variables which meet requirements.
    Get all entry ids which meets the requirements.
    Arguments:
        url (str) :                     URL for GET command (e.g., end with "value")
        variables (list[str,]) :        names of variables to get (empty for all variables)
        requirements (dict{,}) :        required parameters for id
        auth_key (str) :                authorization key
        dtypes (str) :                  data type to extract ('value', 'text', 'version', etc.)
    Returns:
        dictionary (keyed by learner id) of dictionaries (keyed by variables)
        of dictionaries (keyed by dtypes)
    Note:
        If there are multiple values with the same variable name and the same learner id, this function only 
        gets the first value (e.g., "wallet")
    """
    
    data = requests.get(url, headers={'Authorization': 'token '+ str(auth_key)})
    
    if "results" not in data.json():
        print(data)
        raise ValueError("results not found")
    
    values = {}
    
    for entry in data.json()['results']:
            meets_required = True
            if requirements:
                for require_key in requirements:
                    if require_key not in entry or entry[require_key] != requirements[require_key]:
                        meets_required = False
            if meets_required:
                if entry['variable'] in variables or variables == []:
                    if entry['learner'] not in values:
                        values[entry['learner']] = {}           #   create dictionary for empty learner
                    
                    if entry['variable'] not in values[entry['learner']]:
                            values[entry['learner']][entry['variable']] = {}
                    
                    for dtype in dtypes:  
                        if dtype in entry:
                            values[entry['learner']][entry['variable']][dtype] = entry[dtype]
    
    if data.json()['next']:
        url = data.json()['next']
        print(".", end="")
        new_values = get_values_meet_requirements(url, requirements, variables, auth_key, dtypes)

        for learner in new_values:
            if learner in values:       #   if existing learner
                for variable in new_values[learner]:
                    if variable in values[learner]:         # if existing variable
                        for dtype in new_values[learner][variable]:
                            values[learner][variable][dtype] = new_values[learner][variable][dtype]
                    else:
                        values[learner][variable] = {}
                        for dtype in new_values[learner][variable]:
                            values[learner][variable][dtype] = new_values[learner][variable][dtype]
            else:
                values[learner] = {}
                for variable in new_values[learner]:
                    values[learner][variable] = new_values[learner][variable]
    
    return values
                

def delete_request(url, requirements={}, auth_key=""):
    """
    Deletes all entries which meet requirements.
    Arguments:
        url (str) :                     URL for GET command
        requirements (dict{,}) :        required parameters for id (e.g., 'mooclet': 2)
        auth_key (str) :                authorization key
    Returns:
        list of responses from DELETE command (list[dict{}, ])
    """
    data = []
    
    if not requirements:
        trigger = input("WARNING: Empty requirements! This will delete everything. Enter 'y' to proceed...")
        if trigger != 'y':
            raise ValueError("Unintended empty requirements.")
    
    ids = get_ids_meet_requirements(url, requirements, auth_key)
    for id in ids:
        id_url = url + "/" + str(id)
        data.append(requests.delete(id_url, 
                                    headers={'Authorization': 'token '+str(auth_key)}))
        print(".", end="")
    print(str(len(ids))+" entries deleted.")
    
    return data

if __name__ == "__main__":
    
    pass
    
    
    #variable_names = ['age', 'gender', ]