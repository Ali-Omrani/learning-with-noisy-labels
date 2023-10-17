import logging
import re
import json
from tqdm import tqdm
import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import Subset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


def readfile(filename):
    """
        read file
    """
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        line = line.strip()
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = re.compile("\s").split(line)
        sentence.append(splits[0])
        label.append(splits[-1])

    if len(sentence) > 0:
        data.append((sentence, label))
    return data


def read_jsonl_file(filename):
    f = open(filename)
    data = []
    for line in f:
        js = json.loads(line)
        sent1 = js["sentence1"]
        sent2 = js["sentence2"]
        label = js["gold_label"]

        if label not in ["neutral", "entailment", "contradiction"]:
            continue
        data.append((sent1, sent2, label))

    return data


def get_dataset_examples(dataset_name, task):
    if dataset_name.lower() == "mnist":
        t = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        folder = "<data_folder>/MNIST"
        if task == "train":
            mnist = MNIST(folder, download=True, train=True, transform=t)
            return Subset(mnist, range(50000))
        elif task == "dev":
            mnist = MNIST(folder, download=True, train=True, transform=t)
            return Subset(mnist, range(50000, 60000))
        else:
            mnist = MNIST(folder, download=True, train=False, transform=t)
            return mnist
    elif dataset_name.lower() == "cifar10":
        t = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        folder = "<data_folder>/CIFAR10"
        if task == "train":
            cifar10 = CIFAR10(folder, download=True, train=True, transform=t)
            return Subset(cifar10, range(45000))
        elif task == "dev":
            cifar10 = CIFAR10(folder, download=True, train=True, transform=t)
            return Subset(cifar10, range(45000, 50000))
        else:
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            cifar10 = CIFAR10(folder, download=True, train=False, transform=t)
            return cifar10
    elif dataset_name.lower() == "cifar100":
        t = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        folder = "<data_folder>/CIFAR10"
        if task == "train":
            cifar100 = CIFAR100(folder, download=True, train=True, transform=t)
            return Subset(cifar100, range(45000))
        elif task == "dev":
            cifar100 = CIFAR100(folder, download=True, train=True, transform=t)
            return Subset(cifar100, range(45000, 50000))
        else:
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            cifar100 = CIFAR100(folder, download=True, train=False, transform=t)
            return cifar100
    else:
        raise ValueError()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    #TODO
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        try:
            textlist = example.text_a.split(' ')
        except Exception:
            textlist = [example.text]
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []

        if "roberta" in tokenizer.__class__.__name__.lower():
            ntokens.append("<s>")
        else:
            ntokens.append("[CLS]")

        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])

        if "roberta" in tokenizer.__class__.__name__.lower():
            ntokens.append("</s>")
        else:
            ntokens.append("[SEP]")

        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features


def convert_examples_to_features_nli(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 0)}

    features = []
    for (ex_index, example) in tqdm(enumerate(examples), total=len(examples)):
        textlist = example.text_a.split(' ') + ["[SEP]"] + example.text_b.split(' ')
        tokens = []

        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for _ in range(len(token)):
                valid.append(0)
                label_mask.append(0)

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map[example.label])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(0)
        label_mask.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(0)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length
        
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features



def convert_examples_to_features_classification(examples, label_list, max_seq_length, tokenizer, batch_size=128):
    label_map = {label: i for i, label in enumerate(label_list, 0)}

    features = []
    current_batch_text = []

    for (ex_index, example) in tqdm(enumerate(examples), total=len(examples)):
        text = example.text
        current_batch_text.append(text)

        if len(current_batch_text) == batch_size or ex_index == len(examples) - 1:
            # Tokenize the current batch of text
            batch_tokens = tokenizer(current_batch_text, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')

            input_ids = batch_tokens['input_ids']
            input_mask = batch_tokens['attention_mask']
            
            label_ids = [label_map[example.label] for _ in range(len(input_ids))]
            segment_ids = torch.zeros(input_ids.shape, dtype=torch.long)  # Assuming single-segment classification

            valid_ids = torch.zeros(input_ids.shape, dtype=torch.long)
            valid_ids[:, 0] = 1  # Set the first token to 1
            label_mask = valid_ids.clone()
            for i in range(len(input_ids)):
                features.append(
                    InputFeatures(input_ids=input_ids[i],
                                  input_mask=list(input_mask[i]),
                                  segment_ids=list(segment_ids[i]),
                                  label_id=label_ids[i],
                                  valid_ids=list(valid_ids[i]),
                                  label_mask=list(label_mask[i])))
            
            current_batch_text = []  # Clear the current batch

    return features

def convert_examples_to_features_classification_slow(examples, label_list, max_seq_length, tokenizer):

    label_map = {label: i for i, label in enumerate(label_list, 0)}

    features = []
    for (ex_index, example) in tqdm(enumerate(examples), total=len(examples)):
        textlist = example.text.split(' ')
        tokens = []

        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for _ in range(len(token)):
                valid.append(0)
                label_mask.append(0)

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map[example.label])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(0)
        label_mask.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(0)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features

def convert_examples_to_features_ir(examples, label_list):
    features = []
    label_map = {label: i for i, label in enumerate(label_list, 0)}

    for ex in examples:
        features.append(InputFeatures(
            input_ids=ex.image,
            label_id=label_map[ex.label],
            input_mask=[1],
            segment_ids=[1],
            valid_ids=[1],
            label_mask=[1]
        ))

    return features

def get_weights_classes(labels, ratio=1.0):
    classes = set(labels)
    count_per_class = [0] * len(classes)
    for  label in labels:
        count_per_class[label] += 1
    weight_per_class = [0.] * len(classes)
    for i in range(len(classes)):
        weight_per_class[i] = float(len(labels)) / float(count_per_class[i])
    weights = [0] * len(labels)
    for idx, label in enumerate(labels):
        weights[idx] = weight_per_class[label] * ratio if label==1 else weight_per_class[label] * 1-ratio
    return weights
