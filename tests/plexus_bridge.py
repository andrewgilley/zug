"""Actual process-boundary tests; run after `zig build plexus`.

Uses tiny encoded core Wasm fixtures so no extra toolchain dependency is needed.
"""
import hashlib
import json
import pathlib
import subprocess
import tempfile
import unittest

WORKER = pathlib.Path(__file__).resolve().parents[1] / "zig-out/bin/zug-plexus"


def leb(value):
    data = bytearray()
    while True:
        byte = value & 127
        value >>= 7
        data.append(byte | (128 if value else 0))
        if not value:
            return bytes(data)


def section(kind, payload):
    return bytes([kind]) + leb(len(payload)) + payload


def module(code=b"\x41\x07\x0b", memory=None, start=None):
    """() -> i32 export run; optional () -> () start and memory."""
    types = b"\x01\x60\x00\x01\x7f" if start is None else b"\x02\x60\x00\x01\x7f\x60\x00\x00"
    functions = b"\x01\x00" if start is None else b"\x02\x00\x01"
    output = b"\0asm\x01\0\0\0" + section(1, types) + section(3, functions)
    if memory is not None:
        output += section(5, b"\x01\x00" + leb(memory))
    exports = b"\x03run\x00\x00"
    if memory is not None:
        exports += b"\x06memory\x02\x00"
    output += section(7, bytes([1 if memory is None else 2]) + exports)
    if start is not None:
        output += section(8, b"\x01")
    body = b"\x00" + code
    bodies = leb(len(body)) + body
    if start is not None:
        start_body = b"\x00" + start
        bodies += leb(len(start_body)) + start_body
    output += section(10, bytes([1 if start is None else 2]) + bodies)
    return output


class BridgeTests(unittest.TestCase):
    def execute(self, wasm, *, cases=None, fuel=10000, memory=65536, digest=None):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            binary = directory / "worker.wasm"
            binary.write_bytes(wasm)
            envelope = {
                "schema_version": 1,
                "protocol": "plexus-executor/1",
                "module_path": str(binary),
                "module_digest": digest or "sha256:" + hashlib.sha256(wasm).hexdigest(),
                "request": {
                    "export": "run",
                    "result_count": 1,
                    "cases": cases or [{"name": "case", "arguments": [], "memory": None}],
                    "limits": {"fuel_per_case": fuel, "memory_bytes": memory},
                },
            }
            request = directory / "request.json"
            request.write_text(json.dumps(envelope))
            result = subprocess.run([str(WORKER), "execute", str(request)], capture_output=True, text=True, timeout=10)
            return result

    def observation(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        response = json.loads(result.stdout)
        self.assertEqual(response["protocol"], "plexus-executor/1")
        return response["execution"]["cases"][0]

    def test_describe(self):
        result = subprocess.run([str(WORKER), "describe"], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["capabilities"]["max_results"], 1)
        self.assertTrue(json.loads(result.stdout)["capabilities"]["memory_bytes"])

    def test_scalar_digest_and_protocol(self):
        wasm = module()
        result = self.execute(wasm)
        self.assertEqual(self.observation(result)["outcome"], {"kind": "returned", "values": [7]})
        self.assertEqual(json.loads(result.stdout)["module_digest"], "sha256:" + hashlib.sha256(wasm).hexdigest())
        rejected = self.execute(wasm, digest="sha256:" + "0" * 64)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(rejected.stdout, "")
        self.assertIn("ModuleDigestMismatch", rejected.stderr)

    def test_fuel_stops_infinite_loop_in_both_stages(self):
        loop = b"\x03\x40\x0c\x00\x0b\x00\x0b"
        observed = self.observation(self.execute(module(loop), fuel=37))
        self.assertEqual(observed["outcome"], {"kind": "fuel_exhausted", "stage": "invocation"})
        self.assertEqual(observed["fuel_consumed"], 37)
        observed = self.observation(self.execute(module(start=loop), fuel=37))
        self.assertEqual(observed["outcome"], {"kind": "fuel_exhausted", "stage": "initialization"})

    def test_actual_memory_input_and_fresh_instances(self):
        load = b"\x41\x00\x2d\x00\x00\x0b"
        result = self.execute(module(load, memory=1), cases=[
            {"name": "write", "arguments": [], "memory": {"export": "memory", "offset": 0, "utf8": "é"}},
            {"name": "fresh", "arguments": [], "memory": None},
        ])
        self.assertEqual(self.observation(result)["outcome"]["values"], [195])
        self.assertEqual(json.loads(result.stdout)["execution"]["cases"][1]["outcome"]["values"], [0])

    def test_binary_memory_preserves_every_byte_and_case_isolation(self):
        load_word = b"\x41\x07\x28\x00\x00\x0b"
        result = self.execute(module(load_word, memory=1), cases=[
            {"name": "binary", "arguments": [], "memory": {"export": "memory", "offset": 7, "bytes": [0, 255, 128, 127]}},
            {"name": "fresh", "arguments": [], "memory": None},
        ])
        self.assertEqual(self.observation(result)["outcome"]["values"], [0x7f80ff00])
        self.assertEqual(json.loads(result.stdout)["execution"]["cases"][1]["outcome"]["values"], [0])

    def test_memory_payload_rejects_ambiguous_and_invalid_types(self):
        for payload in [
            {}, {"utf8": "x", "bytes": [120]}, {"bytes": None}, {"bytes": "text"},
            {"bytes": [256]}, {"bytes": [-1]}, {"bytes": [1.5]}, {"bytes": [True]},
            {"utf8": [120]}, {"utf8": None}, {"bytes": [], "surprise": 1},
        ]:
            with self.subTest(payload=payload):
                memory = {"export": "memory", "offset": 0, **payload}
                result = self.execute(module(memory=1), cases=[{"name": "invalid", "arguments": [], "memory": memory}])
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_memory_limits_count_bytes_and_reject_offset_overflow(self):
        for payload in [{"utf8": "é"}, {"bytes": [0, 255]}]:
            cases = [{"name": "edge", "arguments": [], "memory": {"export": "memory", "offset": 65534, **payload}}]
            self.assertEqual(self.observation(self.execute(module(memory=1), cases=cases))["outcome"]["kind"], "returned")
            for offset in [65535, 2**64 - 1, -1, "0"]:
                cases[0]["memory"]["offset"] = offset
                result = self.execute(module(memory=1), cases=cases)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
        cases = [{"name": "large", "arguments": [], "memory": {"export": "memory", "offset": 0, "bytes": [0] * (1024 * 1024 + 1)}}]
        result = self.execute(module(memory=32), memory=2 * 1024 * 1024, cases=cases)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("InvalidMemoryInput", result.stderr)

    def test_memory_zero_minimum_and_growth_cap(self):
        self.assertEqual(self.observation(self.execute(module(b"\x3f\x00\x0b", memory=0)))["outcome"]["values"], [0])
        grow = module(b"\x41\x01\x40\x00\x0b", memory=1)
        self.assertEqual(self.observation(self.execute(grow))["outcome"]["values"], [-1])
        self.assertEqual(self.observation(self.execute(grow, memory=131072))["outcome"]["values"], [1])
        result = self.execute(module(memory=2))
        self.assertEqual(json.loads(result.stdout)["execution"]["kind"], "unsupported")

    def test_trap_and_input_failure_are_different(self):
        trapped = self.observation(self.execute(module(b"\x00\x0b")))
        self.assertEqual(trapped["outcome"]["code"], "UnreachableCodeReached")
        result = self.execute(module(memory=1), cases=[{"name": "bad", "arguments": [], "memory": {"export": "missing", "offset": 0, "utf8": "x"}}])
        self.assertEqual(self.observation(result)["outcome"]["kind"], "failed")
        self.assertEqual(self.observation(result)["outcome"]["stage"], "input")

    def test_unsupported_memory_flags_are_rejected(self):
        ordinary = module(memory=1)
        declaration = section(5, b"\x01\x00\x01")
        for flags in [2, 4, 8]:
            unusual = ordinary.replace(declaration, section(5, bytes([1, flags, 1])))
            result = self.execute(unusual)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["execution"], {"kind": "unsupported", "reason": "UnsupportedLimits"})

    def test_invalid_resource_limits_rejected(self):
        for change in [{"fuel": 0}, {"fuel": 10000001}, {"memory": 1}]:
            result = self.execute(module(), **change)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
