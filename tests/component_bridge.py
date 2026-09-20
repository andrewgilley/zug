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


# Exact binary emitted for Plexus's scan WAT: list<u8> -> list<u8>, with a
# return area and a post-return that counts its own release.
SCAN = bytes.fromhex(
    "0061736d0d0001000187030061736d0100000001130360047f7f7f7f017f6002"
    "7f7f017f60017f000304030001020503010001060b027f0141080b7f0141000b"
    "073104066d656d6f727902000c636162695f7265616c6c6f630000047363616e"
    "00010e636162695f706f73745f7363616e00020a8601032001017f2300200241"
    "016b6a200241016b417f73712104200420036a240020040b5901047f41004100"
    "410120011000210202400340200420014f0d012005200020046a2d00006a2105"
    "200220046a20053a0000200441016a21040c000b0b4100410041044108100021"
    "03200320023602002003200136020420030b0900230141016a24010b00930104"
    "6e616d650005047363616e010b010008616c6c6f63617465024f03000500036f"
    "6c6401086f6c645f73697a650205616c69676e030473697a6504037074720106"
    "000370747201036c656e02036f75740304617265610405696e64657805037375"
    "6d020100046172656103140101020004646f6e6501096e6578742d6279746507"
    "110200046e657874010872656c65617365640204010000000630030002010006"
    "6d656d6f7279000001000c636162695f7265616c6c6f63000001000e63616269"
    "5f706f73745f7363616e070e02707d4001056279746573000000060a01000001"
    "00047363616e080c0100000203030004000501010b0a0100047363616e010000"
    "006d0e636f6d706f6e656e742d6e616d65011900000200077265616c6c6f6301"
    "0b706f73742d72657475726e010b00020100066d656d6f727901090011010004"
    "7363616e0113001201000e696d706c656d656e746174696f6e01140302000562"
    "7974657301097363616e2d74797065"
)

# Exact binary emitted for Plexus's summarize WAT: string -> record, whose
# record type the component must export before it can answer with it.
SUMMARIZE = bytes.fromhex(
    "0061736d0d00010001f7020061736d0100000001130360047f7f7f7f017f6002"
    "7f7f017f60017f0003040300010205030100010606017f0141080b073b04066d"
    "656d6f727902000c636162695f7265616c6c6f6300000973756d6d6172697a65"
    "000113636162695f706f73745f73756d6d6172697a6500020a7a032001017f23"
    "00200241016b6a200241016b417f73712104200420036a240020040b5401037f"
    "4101210402400340200320014f0d012004200020036a2d0000410a466a210420"
    "0341016a21030c000b0b410041004104410c1000210220022001360200200220"
    "04360204200220002d00003a000820020b02000b008b01046e616d65000a0973"
    "756d6d6172697a65010b010008616c6c6f63617465024c03000500036f6c6401"
    "086f6c645f73697a650205616c69676e030473697a6504037074720105000370"
    "747201036c656e0204617265610305696e64657804056c696e65730201000461"
    "72656103140101020004646f6e6501096e6578742d6279746507070100046e65"
    "787402040100000006350300020100066d656d6f7279000001000c636162695f"
    "7265616c6c6f630000010013636162695f706f73745f73756d6d6172697a6507"
    "1801720305627974657379056c696e6573790566697273747d0b0d0100077375"
    "6d6d617279030000070b0140010474657874730001060f01000001000973756d"
    "6d6172697a65080c0100000203030004000501020b0f01000973756d6d617269"
    "7a650100000089010e636f6d706f6e656e742d6e616d65011900000200077265"
    "616c6c6f63010b706f73742d72657475726e010b00020100066d656d6f727901"
    "0e001101000973756d6d6172697a650113001201000e696d706c656d656e7461"
    "74696f6e012b0303000773756d6d617279010e73756d6d6172792d6578706f72"
    "74020e73756d6d6172697a652d74797065"
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


# checksum(ptr, len) over a list the caller placed in the component's memory.
CHECKSUM_LOOP = bytes.fromhex(
    "01027f"          # two i32 locals: the running sum and the index
    "0240"            # block
    "0340"            # loop
    "2003" "2001" "4f" "0d01"      # if index >= length, leave the block
    "2002" "2000" "2003" "6a" "2d0000" "6a" "2102"  # sum += memory[ptr + index]
    "2003" "4101" "6a" "2103"      # index += 1
    "0c00" "0b" "0b"  # repeat, end loop, end block
    "2002" "0b")      # return the sum
BUMP = bytes.fromhex("0041100b")      # cabi_realloc: always the same free space
HOSTILE = bytes.fromhex("00417f0b")   # cabi_realloc: a pointer outside memory


def list_core_module(realloc=BUMP):
    """A core module whose memory and allocator the canonical ABI can use."""
    types = b"\x02" + b"\x60\x02\x7f\x7f\x01\x7f" + b"\x60\x04" + b"\x7f" * 4 + b"\x01\x7f"
    binary = b"\0asm\x01\0\0\0" + section(1, types)
    binary += section(3, b"\x02\x00\x01")
    binary += section(5, b"\x01\x00\x01")
    binary += section(7, b"\x03" + name("memory") + b"\x02\x00"
                      + name("cabi_realloc") + b"\x00\x01"
                      + name("checksum") + b"\x00\x00")
    bodies = [CHECKSUM_LOOP, realloc]
    binary += section(10, leb(len(bodies)) + b"".join(leb(len(b)) + b for b in bodies))
    return binary


def list_component(realloc=BUMP, *, options=b"\x02\x03\x00\x04\x00"):
    """A component lifting checksum(list<u8>) -> u32 through its own memory."""
    binary = b"\0asm\x0d\0\x01\0"
    binary += section(1, list_core_module(realloc))
    binary += section(2, b"\x01\x00\x00\x00")
    aliases = (b"\x00\x02\x01\x00" + name("memory")
               + b"\x00\x00\x01\x00" + name("cabi_realloc")
               + b"\x00\x00\x01\x00" + name("checksum"))
    binary += section(6, b"\x03" + aliases)
    binary += section(7, b"\x02" + b"\x70\x7d"
                      + b"\x40\x01" + name("bytes") + b"\x00\x00\x79")
    binary += section(8, b"\x01\x00\x00\x01" + options + b"\x01")
    binary += section(11, b"\x01\x00" + name("checksum") + b"\x01\x00\x00")
    return binary


class ComponentBridgeTests(unittest.TestCase):
    def execute(self, binary=None, *, export="checksum", cases=None, fuel=10000,
                memory=65536, digest=None, change=None,
                profile="fixed-list-u8-u32/1"):
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
                    "profile": profile, "export": export,
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

    @staticmethod
    def list_case(label, value):
        return {"name": label, "arguments": [{"type": "list<u8>", "value": value}]}

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

    def test_a_list_answer_returns_through_the_components_return_area(self):
        cases = [self.list_case("ascending", [1, 2, 3, 4]),
                 self.list_case("wraps-at-a-byte", [200, 100, 50])]
        observations = self.observations(
            self.execute(SCAN, export="scan", cases=cases, profile="list-u8-list-u8/1"))
        self.assertEqual([case["outcome"]["values"] for case in observations],
                         [[{"type": "list<u8>", "value": [1, 3, 6, 10]}],
                          [{"type": "list<u8>", "value": [200, 44, 94]}]])

    def test_a_scalar_profile_does_not_fit_a_list_answer(self):
        cases = [self.list_case("ascending", [1, 2, 3, 4])]
        self.unsupported(self.execute(SCAN, export="scan", cases=cases, profile="list-u8-u32/1"))
        self.unsupported(self.execute(list_component(), cases=cases, profile="list-u8-list-u8/1"))

    def test_a_string_argument_and_a_record_answer(self):
        cases = [{"name": "one-line",
                  "arguments": [{"type": "string", "value": "hello"}]},
                 {"name": "multi-byte",
                  "arguments": [{"type": "string", "value": "h\u00e9llo"}]}]
        observations = self.observations(
            self.execute(SUMMARIZE, export="summarize", cases=cases, profile="string-record/1"))
        self.assertEqual([case["outcome"]["values"] for case in observations],
                         [[{"type": "record", "value": {"bytes": 5, "lines": 1, "first": 104}}],
                          [{"type": "record", "value": {"bytes": 6, "lines": 1, "first": 104}}]])

    def test_a_string_is_written_as_text_and_must_be_valid(self):
        for value in [[104, 105], 5, None, "", "x" * 257]:
            with self.subTest(value=value):
                self.rejected(self.execute(SUMMARIZE, export="summarize", profile="string-record/1",
                                           cases=[{"name": "invalid",
                                                   "arguments": [{"type": "string", "value": value}]}]))
        # A list of bytes is not a string, whichever way round.
        self.rejected(self.execute(SUMMARIZE, export="summarize", profile="string-record/1",
                                   cases=[self.list_case("mistyped", [1, 2, 3])]))
        self.unsupported(self.execute(SUMMARIZE, export="summarize", profile="list-u8-list-u8/1",
                                      cases=[self.list_case("mistyped", [1, 2, 3])]))

    def test_list_capability_is_advertised(self):
        result = subprocess.run([str(WORKER), "describe"], capture_output=True, text=True, check=True)
        capabilities = json.loads(result.stdout)["capabilities"]
        self.assertTrue(capabilities["component_list"])
        self.assertTrue(capabilities["component_list_result"])
        self.assertTrue(capabilities["component_record"])

    def test_list_of_unknown_length_travels_through_component_memory(self):
        cases = [self.list_case("single", [7]),
                 self.list_case("longer-than-flattened", list(range(1, 11))),
                 self.list_case("high-bytes", [255] * 8)]
        observations = self.observations(
            self.execute(list_component(), cases=cases, profile="list-u8-u32/1"))
        self.assertEqual([case["outcome"]["values"] for case in observations],
                         [[{"type": "u32", "value": 7}],
                          [{"type": "u32", "value": 55}],
                          [{"type": "u32", "value": 2040}]])

    def test_requested_profile_must_be_the_component_shape(self):
        # Neither direction is a behavioral answer: the request does not fit.
        self.unsupported(self.execute(list_component(),
                                      cases=[self.case("sum", [1, 2, 3, 4])]))
        self.unsupported(self.execute(REFERENCE, profile="list-u8-u32/1",
                                      cases=[self.list_case("sum", [1, 2, 3, 4])]))

    def test_the_components_own_allocator_decides_where_bytes_go(self):
        # A pointer outside memory fails that case; the host never writes blind.
        case = self.list_case("refused", [1, 2, 3])
        observation = self.observations(
            self.execute(list_component(HOSTILE), cases=[case], profile="list-u8-u32/1"))[0]
        self.assertEqual(observation["outcome"]["kind"], "failed", observation)
        self.assertEqual(observation["outcome"]["stage"], "invocation")

        # A lift that declares no memory cannot carry a list at all.
        self.unsupported(self.execute(list_component(options=b"\x01\x04\x00"),
                                      cases=[case], profile="list-u8-u32/1"))

    def test_list_arguments_are_bounded_and_typed(self):
        for value in [[], [0] * 257, [0, 0, -1], "abcd"]:
            with self.subTest(value=value):
                self.rejected(self.execute(list_component(), profile="list-u8-u32/1",
                                           cases=[self.list_case("invalid", value)]))
        # The type string belongs to the profile, in both directions.
        self.rejected(self.execute(list_component(), profile="list-u8-u32/1",
                                   cases=[self.case("mistyped", [1, 2, 3, 4])]))
        self.rejected(self.execute(cases=[self.list_case("mistyped", [1, 2, 3, 4])]))

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
