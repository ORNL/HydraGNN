from __future__ import annotations


import os
import zlib
from pathlib import Path

import lmdb
import numpy as np

from ase import Atoms
from ase.db.core import Database, now, ops
from ase.db.row import AtomsRow
from ase.io.jsonio import decode, encode
from ase.calculators.calculator import all_properties

# These are special keys in the ASE LMDB that hold
# nextid and deleted ids. The number of ids stored in this
# database is len(range(1,nextid)) - len(deleted_ids)
RESERVED_KEYS = ["nextid", "deleted_ids", "metadata"]


class LMDBDatabase(Database):
    def __init__(
        self,
        filename: str | Path | None = None,
        create_indices: bool = True,
        use_lock_file: bool = True,
        serial: bool = True,
        readonly: bool = False,
        readahead: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        For the most part, this is identical to the other ASE db initializatoins

        LMDB-specific:
            readonly (bool): whether to open lmdb in read-only mode, useful for fast AI/ML training
            readahead (bool): whether to use the LMDB readahead option

        ASE database options:
            use_lock_file: whether to use the ASE lock file implementation
            serial: whether to use the ASE parallelization scheme
        """
        super().__init__(
            Path(filename),
            create_indices,
            use_lock_file,
            serial,
            *args,
            **kwargs,
        )

        # Add a readonly mode for when we're only training
        # to make sure there's no parallel locks
        self.readonly = readonly

        # Toggle readahead (good for seq read, bad for random read)
        self.readahead = readahead

        if self.readonly:
            # Open a new env
            self.env = lmdb.open(
                str(self.filename),
                subdir=False,
                meminit=False,
                map_async=True,
                readonly=True,
                lock=False,
                readahead=self.readahead,
            )

        else:
            # Open a new env with write access
            self.env = lmdb.open(
                str(self.filename),
                map_size=2 ** 41,  # 2Tb max size, typical for older FS
                subdir=False,
                meminit=False,
                map_async=True,
                readahead=self.readahead,
            )

        # Load all ids based on keys in the DB.
        self.ids = []
        self.deleted_ids = set()
        self._load_ids()

    def __enter__(self):
        # If we're in a context manager, get a transaction that we
        # can use repeatedly.
        self.txn = self.env.begin(write=not self.readonly)
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:
        # We're leaving a context manager, so clean up the transaction
        # that has any changes!
        self.txn.commit()
        del self.txn
        self.close()

    def close(self) -> None:
        # Close the lmdb environment
        self.env.close()

    def __del__(self) -> None:
        # If we delete this object, make sure we also close the lmdb env
        self.close()

    def _write(
        self,
        atoms: Atoms | AtomsRow,
        key_value_pairs: dict,
        data: dict | None,
        idx: int | None = None,
    ) -> None:
        # Mostly following the jsondb implementation
        Database._write(self, atoms, key_value_pairs, data)

        mtime = now()

        if isinstance(atoms, AtomsRow):
            row = atoms
        else:
            row = AtomsRow(atoms)
            row.ctime = mtime
            row.user = os.getenv("USER")

        dct = {}
        for key in row.__dict__:
            if key[0] == "_" or key in row._keys or key == "id":
                continue
            dct[key] = row[key]

        dct["mtime"] = mtime

        if key_value_pairs:
            dct["key_value_pairs"] = key_value_pairs

        if data:
            dct["data"] = data
        else:
            dct["data"] = {}

        constraints = row.get("constraints")
        if constraints:
            dct["constraints"] = [constraint.todict() for constraint in constraints]

        # json doesn't like Cell objects, so make it an array
        dct["cell"] = np.asarray(dct["cell"])

        # Get a transaction (or re-use one if we're in a context manager)
        txn = self._get_txn(write=True)

        try:
            if idx is None:
                # We didn't specify an index up front, so we'll just the nextid
                # and increment it
                idx = self._nextid
                nextid = idx + 1
            else:
                # We have an index already, so let's make sure there's nothing already
                # at that row
                data = txn.get(encode_key(f"{idx}"))
                assert data is not None

            # Add the new entry, compressing and encoding the dictionary as json
            txn.put(
                encode_key(f"{idx}"),
                encode_object(dct, compress=True, json_encode=True),
            )
            # only append if idx is not in ids
            if idx not in self.ids:
                self.ids.append(idx)
                txn.put(
                    encode_key("nextid"),
                    encode_object(nextid, compress=True, json_encode=True),
                )
        finally:
            # commit the txn if we're not in a context manager
            self._flush_txn(txn)

        # check if id is in removed ids and remove accordingly
        if idx in self.deleted_ids:
            self.deleted_ids.remove(idx)
            self._write_deleted_ids()

        return idx

    def _update(
        self,
        idx: int,
        key_value_pairs: dict | None = None,
        data: dict | None = None,
    ):
        # Get the existing row using the specific index
        row = self._get_row(idx, include_data=True)

        # update the data or key/value pairs for the row
        if data is not None or key_value_pairs is not None:
            self._write(atoms=row, idx=idx, key_value_pairs=key_value_pairs, data=data)

    def _write_deleted_ids(self):
        # Get a transaction to write with
        txn = self._get_txn(write=True)
        try:
            # Write the current list of deleted ids
            txn.put(
                encode_key("deleted_ids"),
                encode_object(list(self.deleted_ids), compress=True, json_encode=True),
            )
        finally:
            self._flush_txn(txn)

    def _get_txn(self, write):
        # if we already have a txn attached to this object (eg we made one with a context manager),
        # use that. Otherwise, make a new transaction to use
        if hasattr(self, "txn"):
            return self.txn
        else:
            return self.env.begin(write=write)

    def _flush_txn(self, txn):
        # If we have a txn attached to this object, we're in a context manager, so don't commit.
        # otherwise, commit this so the txn gets cleaned up!
        if not hasattr(self, "txn"):
            txn.commit()

    def delete(self, ids: list[int]) -> None:
        # Get a transaction to write against
        txn = self._get_txn(write=True)
        try:
            # Delete the data, remove from the ids, and add to deleted ids, one at a time
            for idx in ids:
                txn.delete(encode_key(f"{idx}"))
                self.ids.remove(idx)
                self.deleted_ids.add(idx)
        finally:
            # Commit the deletes, and write the deleted ids
            self._flush_txn(txn)
            self._write_deleted_ids()

    def _get_row(self, idx: int, include_data: bool = True):
        # Mostly following the jsondb implementation
        if idx is None:
            assert len(self.ids) == 1
            idx = self.ids[0]
        txn = self._get_txn(write=False)
        try:
            row_data = txn.get(encode_key(f"{idx}"))
        finally:
            self._flush_txn(txn)
        if row_data is not None:
            dct = decode_bytestream(row_data, decompress=True, json_decode=True)
        else:
            raise KeyError(f"Id {idx} missing from the database!")
        if not include_data:
            dct.pop("data", None)

        # Anything that's a calculator property should be an array if it's more than a number
        # This is important for things like fmax, which assume forces is an array (and will fail if it's
        # a list of lists)
        for key in all_properties:
            if key in dct and isinstance(dct[key], list):
                dct[key] = np.array(dct[key])

        dct["id"] = idx

        return AtomsRow(dct)

    def _get_row_by_index(self, index: int, include_data: bool = True):
        """Auxiliary function to get the ith entry, rather than a specific id
        In AI/ML training, we often want to just grab the ith entry in the db, rather
        than trying to find the entry ahead of time. We can then move very fast, pulling from
        range(0,len(db))
        """
        return self._get_row(self.ids[index])

    def _select(
        self,
        keys,
        cmps: list[tuple[str, str, str]],
        explain: bool = False,
        verbosity: int = 0,
        limit: int | None = None,
        offset: int = 0,
        sort: str | None = None,
        include_data: bool = True,
        columns: str = "all",
    ):
        # Mostly following the jsondb implementation

        if explain:
            yield {"explain": (0, 0, 0, "scan table")}
            return

        # If we're just trying to find a row with a specific index,
        # we can do that very quickly and without doing a linear scan!
        if len(keys) == 0 and (
            len(cmps) == 1 and cmps[0][0] == "id" and cmps[0][1] == "="
        ):
            yield self._get_row(cmps[0][2], include_data=include_data)
            return

        if sort is not None and len(sort) > 0:
            if sort[0] == "-":
                reverse = True
                sort = sort[1:]
            else:
                reverse = False

            rows = []
            missing = []
            for row in self._select(keys, cmps):
                key = row.get(sort)
                if key is None:
                    missing.append((0, row))
                else:
                    rows.append((key, row))

            rows.sort(reverse=reverse, key=lambda x: x[0])
            rows += missing

            if limit:
                rows = rows[offset : offset + limit]
            for _, row in rows:
                yield row
            return

        if not limit:
            limit = -offset - 1

        cmps = [(key, ops[op], val) for key, op, val in cmps]
        n = 0
        for idx in self.ids:
            if n - offset == limit:
                return
            row = self._get_row(idx, include_data=include_data)

            for key in keys:
                if key not in row:
                    break
            else:
                for key, op, val in cmps:
                    if isinstance(key, int):
                        value = np.equal(row.numbers, key).sum()
                    else:
                        value = row.get(key)
                        if key == "pbc":
                            assert op in [ops["="], ops["!="]]
                            value = "".join("FT"[x] for x in value)
                    if value is None or not op(value, val):
                        break
                else:
                    if n >= offset:
                        yield row
                    n += 1

    @property
    def _nextid(self):
        """Get the id of the next row to be written"""
        # Get the nextid
        txn = self._get_txn(write=False)
        try:
            nextid_data = txn.get(encode_key("nextid"))
            return (
                decode_bytestream(nextid_data, decompress=True, json_decode=True)
                if nextid_data
                else 1
            )
        finally:
            self._flush_txn(txn)

    def count(self, selection=None, **kwargs) -> int:
        """Count rows.

        See the select() method for the selection syntax.  Use db.count() or
        len(db) to count all rows.
        """
        if selection is not None:
            n = 0
            for _row in self.select(selection, **kwargs):
                n += 1
            return n
        else:
            # Fast count if there's no queries! Just get number of ids
            return len(self.ids)

    def _load_ids(self) -> None:
        """Load ids from the DB

        Since ASE db ids are mostly 1-N integers, but can be missing entries
        if ids have been deleted. To save space and operating under the assumption
        that there will probably not be many deletions in most FAIR chemistry datasets,
        we just store the deleted ids.

        This is a bad idea if you repeatedly read/write/delete to the same DB file
        """

        # Load the deleted ids
        txn = self._get_txn(write=False)
        try:
            deleted_ids_data = txn.get(encode_key("deleted_ids"))
        finally:
            self._flush_txn(txn)

        if deleted_ids_data is not None:
            self.deleted_ids = set(
                decode_bytestream(deleted_ids_data, decompress=True, json_decode=True)
            )

        # Reconstruct the full id list
        self.ids = [i for i in range(1, self._nextid) if i not in self.deleted_ids]

    def get_all_key_names(self):
        """
        We don't have a choice but to loop over all rows to find all keys. This could be done
        faster if we stored this list of keys in the db too.
        """
        return set.union(
            *[set(self._get_row(id).key_value_pairs.keys()) for id in self.ids]
        )

    @property
    def metadata(self):
        """
        metadata is a helpful json-encoded dict to help us understand the context
        of this database
        """
        txn = self._get_txn(write=False)
        try:
            metadata_stream = txn.get(encode_key("metadata"))
        finally:
            self._flush_txn(txn)
        if metadata_stream is not None:
            return decode_bytestream(metadata_stream, decompress=True, json_decode=True)
        else:
            return {}

    @metadata.setter
    def metadata(self, value: dict):
        txn = self._get_txn(write=True)
        try:
            txn.put(
                encode_key("metadata"),
                encode_object(value, compress=True, json_encode=True),
            )
        finally:
            self._flush_txn(txn)


def encode_key(input):
    """
    Default encoding for all keys in the db
    """
    return input.encode("ascii")


def encode_object(input, compress=True, json_encode=False):
    """
    Encoding (optionally compressed json) using the ase encoder and zlib
    """
    if json_encode:
        input = encode(input)
    if compress:
        return zlib.compress(bytes(input, "utf-8"))
    else:
        return bytes(input, "utf-8")


def decode_bytestream(bytestream, decompress=True, json_decode=False):
    """
    Decoding (optionally compressed json) using the ase encoder and zlib
    """
    if decompress:
        decoded_bytestream = zlib.decompress(bytestream).decode("utf-8")
    else:
        decoded_bytestream = bytestream.decode("utf-8")
    if json_decode:
        return decode(decoded_bytestream)
    else:
        return decoded_bytestream
