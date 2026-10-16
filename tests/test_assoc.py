from neuromem.assoc import WillshawMemory
from neuromem.codes import SparseCoder

TEXTS = ["the billing service charges customers monthly", "search indexes are rebuilt every night",
         "checkout uses the payments gateway", "the ledger reconciles transactions daily",
         "media uploads are resized into thumbnails", "inventory counts are synced hourly"]


def test_pattern_completion_recalls_from_fragment_with_fixed_ram():
    coder = SparseCoder(neurons=2048)
    net = WillshawMemory(neurons=2048, id_units=1024)
    ram = net.ram_bytes()
    for i, t in enumerate(TEXTS):
        net.store(i, coder(t))
    assert net.ram_bytes() == ram                               # weights don't grow with stored memories
    assert net.recall(coder("search nightly indexes"), range(len(TEXTS)))[0] == 1
    assert net.recall(coder("thumbnails resized"), range(len(TEXTS)))[0] == 4
    full = coder(TEXTS[2])
    assert set(full) <= set(net.complete(full[: len(full) // 2]))  # half a pattern completes to the whole
