"""QuantBot — data-fusion forex research/demo-trading system.

Demo/paper-first by design: no component here places a live order unless the
operator explicitly flips `broker.allow_live` *and* the promotion gate
(`quantbot.ops.gate`) has been evaluated on a real demo track record.
"""

__version__ = "0.1.0"
