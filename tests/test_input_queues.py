"""Every requested-album queue is read before any automatic-album queue.

    python -m pytest tests/test_input_queues.py -v
"""
import logging
from collections import deque

import pytest
from ptinfra.pt_queue import QReader

import main


class FakeQueue(QReader):
    def __init__(self, name):
        self.name = name
        self.messages = deque()

    def nextMessage(self, wait_for_message=True):
        return self.messages.popleft() if self.messages else None


def _reader(monkeypatch, environment):
    monkeypatch.setattr(main, 'get_variable', lambda key: environment)
    made = {}

    def factory(name):
        made[name] = FakeQueue(name)
        return made[name]

    return main._get_azure_input_queue(logging.getLogger('test'), factory), made


@pytest.mark.parametrize('environment, requested, automatic', [
    ('dev', ['devaigeneratealbumdto', 'dev3aigeneratealbumdto'],
            ['devaigenerateautoalbumdto', 'dev3aigenerateautoalbumdto']),
    ('production', ['aigeneratealbumdto', 'epaigeneratealbumdto'],
                   ['aigenerateautoalbumdto', 'epaigenerateautoalbumdto']),
    ('sup', ['supaigeneratealbumdto'], ['supaigenerateautoalbumdto']),
])
def test_each_queue_has_an_auto_queue_ranked_below_it(monkeypatch, environment, requested, automatic):
    reader, _ = _reader(monkeypatch, environment)
    assert [[q.name for q in rank] for rank in reader.ranks] == [requested, automatic]


def test_an_auto_album_waits_until_every_requested_queue_is_empty(monkeypatch):
    reader, q = _reader(monkeypatch, 'dev')
    q['devaigenerateautoalbumdto'].messages.extend(['auto1', 'auto2'])
    q['devaigeneratealbumdto'].messages.append('dev1')
    q['dev3aigeneratealbumdto'].messages.append('dev3-1')

    read = [reader.nextMessage(wait_for_message=False) for _ in range(5)]

    assert read[:2] in (['dev1', 'dev3-1'], ['dev3-1', 'dev1'])
    assert read[2:] == ['auto1', 'auto2', None]


def test_a_requested_album_overtakes_waiting_auto_albums(monkeypatch):
    reader, q = _reader(monkeypatch, 'production')
    q['aigenerateautoalbumdto'].messages.extend(['auto1', 'auto2'])
    assert reader.nextMessage(wait_for_message=False) == 'auto1'

    q['epaigeneratealbumdto'].messages.append('ep1')
    assert reader.nextMessage(wait_for_message=False) == 'ep1'
    assert reader.nextMessage(wait_for_message=False) == 'auto2'
