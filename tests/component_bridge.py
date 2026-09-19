"""Component Model bridge process tests; run after `zig build plexus`.

The binary encoder deliberately varies names, indices and guest instructions.
No Rust, wat compiler, archived workspace artifact or expected-answer input is
needed to test the installed worker.
"""
import hashlib
import json
import pathlib
import subprocess
import tempfile
import unittest

from plexus_bridge import WORKER, leb, section


def name(value):
    value = value.encode()
    return leb(len(value)) + value


SUM = bytes.fromhex("200020016a20026a20036a0b")
LOOP = bytes.fromhex("03400c000b000b")
# Exact binary emitted for Plexus's reference checksum WAT (including names).
REFERENCE = bytes.fromhex(
    "0061736d0d00010001480061736d0100000001090160047f7f7f7f017f03020100"
    "070c0108636865636b73756d00000a0f010d00200020016a20026a20036a0b0010"
    "046e616d65000908636865636b73756d020401000000070f02677d044001057661"
    "6c7565000079060e010000010008636865636b73756d08060100000000010b0e01"
    "0008636865636b73756d010000004d0e636f6d706f6e656e742d6e616d65010d00"
    "11010008636865636b73756d0113001201000e696d706c656d656e746174696f6e"
    "0118030200056279746573010d636865636b73756d2d74797065"
)


def core_module(code=SUM, *, export="checksum", memory=None, start=None,
                shifted=False, parameters=4):
    types = b"\x01\x60" + leb(parameters) + b"\x7f" * parameters + b"\x01\x7f"
    if start is not None:
        types = b"\x02" + types[1:] + b"\x60\x00\x00"
    bodies = ([b"\x41\x2a\x0b"] if shifted else []) + [code]
    type_indices = [0] * len(bodies)
    if start is not None:
        bodies.append(start)
        type_indices.append(1)
    binary = b"\0asm\x01\0\0\0" + section(1, types)
    binary += section(3, leb(len(type_indices)) + bytes(type_indices))
    if memory is not None:
        binary += section(5, b"\x01\x00" + leb(memory))
    binary += section(7, b"\x01" + name(export) + b"\x00" + leb(int(shifted)))
    if start is not None:
        binary += section(8, leb(len(bodies) - 1))
    binary += section(10, leb(len(bodies)) + b"".join(
        leb(len(body) + 1) + b"\x00" + body for body in bodies))
    return binary


def component(code=SUM, *, export="checksum", core_export="checksum",
              memory=None, start=None, shifted=False, parameters=4,
              list_length=4, result_type=0x79, alias_index=0, lift_index=0,
              export_index=0, lift_type=1, shifted_component=False):
    binary = b"\0asm\x0d\0\x01\0"
    binary += section(1, core_module(code, export=core_export, memory=memory,
                                     start=start, shifted=shifted,
                                     parameters=parameters))
    binary += section(2, b"\x01\x00\x00\x00")  # instantiate module 0, no imports
    type_prefix = b"\x03\x7d" if shifted_component else b"\x02"
    binary += section(7, type_prefix + b"\x67\x7d" + leb(list_length)
                      + b"\x40\x01" + name("value") + bytes([int(shifted_component), 0, result_type]))
    alias = b"\x00\x00\x01" + leb(alias_index) + name(core_export)
    binary += section(6, (b"\x02" + alias * 2) if shifted_component else b"\x01" + alias)
    canonical = b"\x00\x00" + leb(lift_index) + b"\x00" + leb(lift_type)
    binary += section(8, (b"\x02" + canonical * 2) if shifted_component else b"\x01" + canonical)
    binary += section(11, b"\x01\x00" + name(export) + b"\x01" + leb(export_index) + b"\x00")
    return binary


class ComponentBridgeTests(unittest.TestCase):
    def execute(self, binary=None, *, export="checksum", cases=None, fuel=10000,
                memory=65536, digest=None, change=None):
        if binary is None:
            binary = component()
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            path = directory / "component.wasm"
            path.write_bytes(binary)
            envelope = {
                "schema_version": 1,
                "protocol": "plexus-executor/3",
                "component_path": str(path),
                "component_digest": digest or "sha256:" + hashlib.sha256(binary).hexdigest(),
                "request": {
                    "profile": "fixed-list-u8-u32/1", "export": export,
                    "cases": cases if cases is not None else [self.case("sum", [0, 255, 128, 127])],
                    "limits": {"fuel_per_case": fuel, "memory_bytes": memory},
                    "host_grants": [],
                },
            }
            if change:
                change(envelope)
            request = directory / "request.json"
            request.write_text(json.dumps(envelope))
            return subprocess.run([str(WORKER), "execute", str(request)],
                                  capture_output=True, text=True, timeout=10)

    @staticmethod
    def case(label, value):
        return {"name": label, "arguments": [{"type": "list<u8,4>", "value": value}]}

    def response(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        response = json.loads(result.stdout)
        self.assertEqual(response["protocol"], "plexus-executor/3")
        return response

    def observations(self, result):
        execution = self.response(result)["execution"]
        self.assertEqual(execution["kind"], "observed", execution)
        for case in execution["cases"]:
            self.assertEqual(case["host_trace"], [])
        return execution["cases"]

    def returned(self, result):
        case = self.observations(result)[0]
        self.assertEqual(case["outcome"]["kind"], "returned", case)
        self.assertGreater(case["fuel_consumed"], 0)
        return case["outcome"]["values"]

    def rejected(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, "")

    def unsupported(self, result):
        execution = self.response(result)["execution"]
        self.assertEqual(execution["kind"], "unsupported", execution)
        self.assertTrue(execution["reason"])

    def test_component_capability_is_advertised(self):
        result = subprocess.run([str(WORKER), "describe"], capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(result.stdout)["capabilities"]["component_fixed_list"])

    def test_real_typed_execution_and_digest(self):
        for binary in [REFERENCE, component()]:
            result = self.execute(binary)
            self.assertEqual(self.returned(result), [{"type": "u32", "value": 510}])
            self.assertEqual(self.response(result)["component_digest"], "sha256:" + hashlib.sha256(binary).hexdigest())

    def test_arbitrary_names_indices_and_implementation(self):
        # Subtraction, not checksum; the exported core function is index 1.
        binary = component(bytes.fromhex("200020016b0b"), export="difference",
                           core_export="calculate_difference", shifted=True,
                           shifted_component=True, lift_index=1, lift_type=2, export_index=1)
        cases = [self.case("unsigned-underflow", [0, 1, 240, 37]),
                 self.case("positive", [255, 128, 0, 0])]
        observations = self.observations(self.execute(binary, export="difference", cases=cases))
        self.assertEqual([case["outcome"]["values"] for case in observations],
                         [[{"type": "u32", "value": 4294967295}], [{"type": "u32", "value": 127}]])

    def test_expected_answers_are_never_accepted(self):
        for target in ["envelope", "request", "case", "argument"]:
            with self.subTest(target=target):
                def add_expected(envelope):
                    objects = {"envelope": envelope, "request": envelope["request"],
                               "case": envelope["request"]["cases"][0],
                               "argument": envelope["request"]["cases"][0]["arguments"][0]}
                    objects[target]["expected"] = 510
                self.rejected(self.execute(change=add_expected))

    def test_invalid_typed_arguments(self):
        for value in ["abcd", None, [], [0] * 3, [0] * 5,
                      [0, 0, 0, "1"], [0, 0, 0, 1.0], [0, 0, 0, True],
                      [0, 0, 0, -1], [0, 0, 0, 256]]:
            with self.subTest(value=value):
                self.rejected(self.execute(cases=[self.case("invalid", value)]))
        for value in [[], [{"type": "u32", "value": 1}],
                      [{"type": "list<u8,4>", "value": [0] * 4}] * 2]:
            self.rejected(self.execute(cases=[{"name": "invalid", "arguments": value}]))

    def test_digest_grants_profile_and_limits(self):
        self.rejected(self.execute(digest="sha256:" + "0" * 64))
        for version in ["1", 1.0, True]:
            self.rejected(self.execute(change=lambda envelope: envelope.update({"schema_version": version})))
        for key, value in [("host_grants", ["filesystem"]), ("host_grants", None),
                           ("profile", "unknown"), ("cases", [])]:
            self.rejected(self.execute(change=lambda envelope: envelope["request"].update({key: value})))
        for changes in [{"fuel": 0}, {"fuel": True}, {"fuel": 10000001},
                        {"fuel": "10000"}, {"fuel": 10000.0},
                        {"memory": 0}, {"memory": 67108865},
                        {"memory": "65536"}, {"memory": 65536.0}]:
            self.rejected(self.execute(**changes))

    def test_types_indices_exports_and_trailing_bytes(self):
        for changes in [{"list_length": 3}, {"result_type": 0x7a}, {"parameters": 3},
                        {"alias_index": 1}, {"lift_index": 1}, {"lift_type": 0},
                        {"export_index": 1}]:
            with self.subTest(changes=changes):
                self.unsupported(self.execute(component(**changes)))
        self.unsupported(self.execute(export="missing"))
        self.unsupported(self.execute(component() + b"\xff"))
        self.unsupported(self.execute(component()[:-1]))

    def test_invocation_and_start_are_fuel_bounded(self):
        for stage, binary in [("invocation", component(LOOP)),
                              ("initialization", component(start=LOOP))]:
            with self.subTest(stage=stage):
                case = self.observations(self.execute(binary, fuel=37))[0]
                self.assertEqual(case["outcome"], {"kind": "fuel_exhausted", "stage": stage})
                self.assertEqual(case["fuel_consumed"], 37)

    def test_traps_keep_their_execution_stage(self):
        trap = bytes.fromhex("000b")
        for stage, binary in [("invocation", component(trap)),
                              ("initialization", component(start=trap))]:
            with self.subTest(stage=stage):
                case = self.observations(self.execute(binary))[0]
                self.assertEqual(case["outcome"], {
                    "kind": "failed", "stage": stage,
                    "diagnostic": "UnreachableInstruction",
                })
                self.assertGreater(case["fuel_consumed"], 0)

    def test_memory_cap_and_fresh_instances(self):
        self.unsupported(self.execute(component(memory=2)))
        self.assertEqual(self.returned(self.execute(component(memory=2), memory=131072)),
                         [{"type": "u32", "value": 510}])
        # Increment memory byte zero, then load it: each case must return one.
        increment = bytes.fromhex("410041002d000041016a3a000041002d00000b")
        cases = [self.case("first", [0] * 4), self.case("second", [255] * 4)]
        observed = self.observations(self.execute(component(increment, memory=1), cases=cases))
        self.assertEqual([case["outcome"]["values"][0]["value"] for case in observed], [1, 1])


if __name__ == "__main__":
    unittest.main()
