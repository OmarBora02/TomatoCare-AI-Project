import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations 
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64 



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns 
    Args: 
        path_to_yaml (str): path like input
    
    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type 
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successifully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories
    
    Args:
    path_to_directories (list): list of path of directories 
    ignore_log(bool, optional): ingore if multiple dirs is to be created. Deafualts to False.
    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")



@ensure_annotations
def save_json(path: Path, data: dict):
    """save dictionary data to json file
    
    Args:
    path (Path): path of json file
    data (dict): dictionary data to save
    """
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)
    
    logger.info(f"Saved data to json file: {path}")



@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json file and returns dictionary
    
    Args:
    path (Path): path of json file
    
    Returns:
    ConfigBox: ConfigBox type
    """
    with open(path) as file:
        content =json.load(file)

    logger.info(f"json file loaded successifully from: {path}")
    return content



@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save binary file
    
    Args: 
    data (Any): data to save
    path (Path): path of binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Saved data to binary file: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load binary file and returns data
    
    Args: 
    path (Path): path of binary file
    
    Returns: 
    Any: data loaded from binary file
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded successifully from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """Get size of file in bytes
    
    Args: 
    path (Path): path of file
    
    Returns: 
    str: size of file in bytes
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~{size_in_kb} bytes"


def decodeImage(imgstring, fileName):
    imgdata = base64.basedecode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
