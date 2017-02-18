import json


class Meta(object):
    def __init__(self):
        self.num_train_examples = None
        self.num_val_examples = None
        self.num_test_examples = None

    def save(self, path_to_json_file):
        with open(path_to_json_file, 'w') as f:
            content = {
                'num_examples': {
                    'train': self.num_train_examples,
                    'val': self.num_val_examples,
                    'test': self.num_test_examples
                }
            }
            json.dump(content, f)

    def load(self, path_to_json_file):
        with open(path_to_json_file, 'r') as f:
            content = json.load(f)
            self.num_train_examples = content['num_examples']['train']
            self.num_val_examples = content['num_examples']['val']
            self.num_test_examples = content['num_examples']['test']
