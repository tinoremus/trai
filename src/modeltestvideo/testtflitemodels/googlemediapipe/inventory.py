from dataclasses import dataclass
from typing import List
import requests
from urllib.request import urlretrieve
import xml.etree.ElementTree as ET
import os
LINK = r'https://storage.googleapis.com/mediapipe-models/'


@dataclass()
class InventoryItem:
    Key: str
    Generation: int
    MetaGeneration: int
    LastModified: str
    ETag: str
    Size: int


def get_inventory():

    resp = requests.get(LINK)
    with open('mediapipe.xml', 'wb') as f:
        f.write(resp.content)


def parse_xml() -> List[InventoryItem]:
    tree = ET.parse('mediapipe.xml')
    root = tree.getroot()
    inventory_items = list()
    for element in root:
        if element.tag.endswith('Contents'):
            info = {
                'Key': '',
                'Generation': -1,
                'MetaGeneration': -1,
                'LastModified': '',
                'ETag': '',
                'Size': -1,
            }
            for _item in element:
                if _item.tag.split('}').pop(-1) == 'Key':
                    info['Key'] = _item.text
                elif _item.tag.split('}').pop(-1) == 'Generation':
                    info['Generation'] = _item.text
                elif _item.tag.split('}').pop(-1) == 'MetaGeneration':
                    info['MetaGeneration'] = _item.text
                elif _item.tag.split('}').pop(-1) == 'LastModified':
                    info['LastModified'] = _item.text
                elif _item.tag.split('}').pop(-1) == 'ETag':
                    info['ETag'] = _item.text
                elif _item.tag.split('}').pop(-1) == 'Size':
                    info['Size'] = _item.text
                else:
                    print(_item.tag)
            inventory_items.append(InventoryItem(**info))
    return inventory_items


def download_models(inventory_items: List[InventoryItem]):
    tracking = {}
    for item in inventory_items:
        if not item.Key.endswith('.tflite'):
            continue
        domain = item.Key.split('/').pop(0).replace('_', '')
        precision = item.Key.split('/').pop(2)
        file_name = os.path.basename(item.Key)
        file_name = '{}_{}.tflite'.format(file_name[:file_name.rfind('.')], precision)
        if not os.path.exists(f'./{domain}'):
            os.mkdir(f'./{domain}')
        if domain not in tracking:
            tracking.update({domain: {'etags': []}})

        if item.ETag not in tracking[domain]['etags']:
            url = LINK + item.Key
            destination = os.path.join(f'./{domain}', file_name)
            if not os.path.exists(destination):
                print('Downloading: {} -> {}'.format(url, destination))
                urlretrieve(url, destination)
            else:
                print('Skipping: File already exists.')
        else:
            print('Skipping: Duplicate')
        tracking[domain]['etags'].append(item.ETag)


if __name__ == '__main__':
    get_inventory()
    download_models(parse_xml())
