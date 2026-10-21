import io
import json
import os
import pickle
import random
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import HfFileSystem
from PIL import Image
from tqdm import tqdm

REPO_PATH = "datasets/chenguolin/InstructScene_dataset"

# The two zips we read from. Neither is ever downloaded whole: zip files have
# an index of their members, so single files can be read straight off the Hub.
ROOMS_ZIP = "InstructScene.zip"  # ~5 GB: room renders + per-room metadata
ITEMS_ZIP = "3D-FRONT.zip"  # ~26 GB: mostly 3D meshes; only item preview images are read

# Where downloaded files are stored, relative to the repo root.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "instructscene")

# Rooms listed here (one room_uid per line) are treated as not existing: they
# are never downloaded, sampled or displayed. It lives next to the code, not
# in data/, so it is committed and shared by everyone. Written by
# delete_duplicate_rooms.
EXCLUDED_ROOMS_PATH = os.path.join(os.path.dirname(__file__), "instructscene_excluded_rooms.txt")

ROOM_TYPES = ("bedroom", "diningroom", "livingroom")

# Each room has 8 rendered perspective views (000-007) and a top-down view;
# only one of them is kept per room.
ROOM_VIEWS = tuple(f"{i:03d}" for i in range(8)) + ("topdown",)
DEFAULT_ROOM_VIEW = "000"

# Per-item attributes copied from models_info.pkl. These describe the item
# itself (identical in every room it appears in), so they're stored once per
# item in items.json rather than repeated in every room.
ITEM_FIELDS = (
    "super-category",
    "category",
    "style",
    "theme",
    "material",
    "chatgpt_caption",
    "blip_caption",
    "msft_caption",
)

# How much is fetched per network request when reading from a zip on the Hub.
# Small enough that reading one ~100 KB image doesn't pull megabytes.
REMOTE_BLOCK_SIZE = 256 * 1024


@contextmanager
def _remote_zip(zip_name):
    """
    Open one of the dataset's zips on the Hugging Face Hub for reading,
    without downloading it. Only the zip's index and the members actually
    read are fetched.

    Args:
        zip_name: Filename of the zip in the dataset repo (ROOMS_ZIP or
            ITEMS_ZIP).

    Returns:
        A context manager yielding an open zipfile.ZipFile.
    """
    fs = HfFileSystem()
    with fs.open(f"{REPO_PATH}/{zip_name}", "rb", block_size=REMOTE_BLOCK_SIZE) as f:
        with zipfile.ZipFile(f) as z:
            yield z


def _room_dir_in_zip(room_type, room_uid):
    """
    Path of a room's folder inside InstructScene.zip.

    Args:
        room_type: "bedroom", "diningroom" or "livingroom".
        room_uid: The room's unique ID, e.g.
            "36cf981f-f4fd-41dd-a0c3-52d8f4511855_SecondBedroom-28815".

    Returns:
        str, e.g. "InstructScene/threed_front_bedroom/{room_uid}".
    """
    return f"InstructScene/threed_front_{room_type}/{room_uid}"


def _room_image_in_zip(room_type, room_uid, room_view):
    """
    Path of one rendered view of a room inside InstructScene.zip.

    Args:
        room_type: "bedroom", "diningroom" or "livingroom".
        room_uid: The room's unique ID.
        room_view: Which render to use, one of ROOM_VIEWS.

    Returns:
        str path of the .png inside the zip.
    """
    return f"{_room_dir_in_zip(room_type, room_uid)}/blender_rendered_scene_256/{room_view}.png"


def _item_image_in_zip(jid):
    """
    Path of an item's preview image inside 3D-FRONT.zip.

    Args:
        jid: The item's 3D-FUTURE model ID (called jid in boxes.npz and
            model_id in models_info.pkl).

    Returns:
        str path of the .jpg inside the zip.
    """
    return f"3D-FRONT/3D-FUTURE-model/{jid}/image.jpg"


def _local_room_image_path(cache_dir, room_type, room_uid):
    """
    Where a room's image is stored locally.

    Args:
        cache_dir: Local root directory data is stored under.
        room_type: "bedroom", "diningroom" or "livingroom".
        room_uid: The room's unique ID.

    Returns:
        str, cache_dir/rooms/{room_type}/{room_uid}.png.
    """
    return os.path.join(cache_dir, "rooms", room_type, f"{room_uid}.png")


def _local_item_image_path(cache_dir, jid):
    """
    Where an item's image is stored locally.

    Args:
        cache_dir: Local root directory data is stored under.
        jid: The item's 3D-FUTURE model ID.

    Returns:
        str, cache_dir/items/{jid}.jpg.
    """
    return os.path.join(cache_dir, "items", f"{jid}.jpg")


def _load_excluded_rooms():
    """
    Read the list of excluded rooms (EXCLUDED_ROOMS_PATH).

    Args:
        None.

    Returns:
        Set of room_uids to ignore. Empty if the file doesn't exist.
    """
    if not os.path.exists(EXCLUDED_ROOMS_PATH):
        return set()
    with open(EXCLUDED_ROOMS_PATH) as f:
        lines = (line.strip() for line in f)
        return {line for line in lines if line and not line.startswith("#")}


def _save_excluded_rooms(room_uids):
    """
    Write the list of excluded rooms (EXCLUDED_ROOMS_PATH), sorted so that
    git diffs stay small.

    Args:
        room_uids: Iterable of room_uids to exclude.

    Returns:
        None.
    """
    with open(EXCLUDED_ROOMS_PATH, "w") as f:
        f.write("# Rooms that are ignored everywhere (duplicates of another room).\n")
        f.write("# One room_uid per line. Written by delete_duplicate_rooms.\n")
        for uid in sorted(set(room_uids)):
            f.write(f"{uid}\n")


def _list_remote_rooms(rooms_zip, room_types):
    """
    List every room of the given types in InstructScene.zip, using only the
    zip's index (no room data is read). Excluded rooms are left out.

    Args:
        rooms_zip: Open zipfile.ZipFile of InstructScene.zip.
        room_types: Iterable of room types to include.

    Returns:
        Sorted list of (room_type, room_uid) tuples.
    """
    excluded = _load_excluded_rooms()
    rooms = []
    for name in rooms_zip.namelist():
        parts = name.split("/")
        # A room is identified by its boxes.npz:
        # InstructScene/threed_front_{room_type}/{room_uid}/boxes.npz
        if len(parts) != 4 or parts[3] != "boxes.npz" or not parts[1].startswith("threed_front_"):
            continue
        room_type = parts[1][len("threed_front_"):]
        if room_type in room_types and parts[2] not in excluded:
            rooms.append((room_type, parts[2]))
    return sorted(rooms)


def _parse_room(rooms_zip, room_type, room_uid):
    """
    Read one room's metadata from InstructScene.zip: which items it
    contains, where each one is placed, and each item's attributes.

    Reads boxes.npz (placement of every object) and models_info.pkl (one
    entry per object, in the same order, with the item's ID, category,
    style and captions).

    Args:
        rooms_zip: Open zipfile.ZipFile of InstructScene.zip.
        room_type: "bedroom", "diningroom" or "livingroom".
        room_uid: The room's unique ID.

    Returns:
        A (room, items) tuple:
          - room: dict with "room_uid", "room_type", "scene_type" (finer
            label, e.g. "secondbedroom") and "objects", a list with one
            dict per object: "jid", "translation", "size", "angle".
          - items: dict mapping each jid in the room to its attributes
            (the ITEM_FIELDS).
    """
    room_dir = _room_dir_in_zip(room_type, room_uid)
    boxes = np.load(io.BytesIO(rooms_zip.read(f"{room_dir}/boxes.npz")))
    models_info = pickle.loads(rooms_zip.read(f"{room_dir}/models_info.pkl"))

    jids = [str(jid) for jid in boxes["jids"]]
    if [info["model_id"] for info in models_info] != jids:
        raise ValueError(f"boxes.npz and models_info.pkl disagree on the items in room {room_uid}")

    objects = [
        {
            "jid": jid,
            "translation": translation.tolist(),
            "size": size.tolist(),
            "angle": float(angle[0]),
        }
        for jid, translation, size, angle in zip(jids, boxes["translations"], boxes["sizes"], boxes["angles"])
    ]
    room = {
        "room_uid": room_uid,
        "room_type": room_type,
        "scene_type": str(boxes["scene_type"]),
        "objects": objects,
    }
    items = {info["model_id"]: {field: info.get(field) for field in ITEM_FIELDS} for info in models_info}
    return room, items


def _load_local_metadata(cache_dir):
    """
    Load the locally stored room and item metadata, if any has been
    downloaded. Excluded rooms are left out.

    Args:
        cache_dir: Local root directory data is stored under.

    Returns:
        A (rooms, items) tuple of dicts, keyed by room_uid and jid. Both
        are empty if nothing has been downloaded yet.
    """
    rooms_path = os.path.join(cache_dir, "metadata", "rooms.json")
    items_path = os.path.join(cache_dir, "metadata", "items.json")
    if not (os.path.exists(rooms_path) and os.path.exists(items_path)):
        return {}, {}

    excluded = _load_excluded_rooms()
    with open(rooms_path) as f:
        rooms = {room["room_uid"]: room for room in json.load(f) if room["room_uid"] not in excluded}
    with open(items_path) as f:
        items = json.load(f)
    return rooms, items


def _save_local_metadata(cache_dir, rooms, items):
    """
    Write the room and item metadata to cache_dir/metadata/.

    Args:
        cache_dir: Local root directory data is stored under.
        rooms: Dict of room records keyed by room_uid.
        items: Dict of item attributes keyed by jid.

    Returns:
        None.
    """
    metadata_dir = os.path.join(cache_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    with open(os.path.join(metadata_dir, "rooms.json"), "w") as f:
        json.dump([rooms[uid] for uid in sorted(rooms)], f)
    with open(os.path.join(metadata_dir, "items.json"), "w") as f:
        json.dump({jid: items[jid] for jid in sorted(items)}, f)


def _run_on_remote_zip(zip_name, tasks, worker, num_workers, desc):
    """
    Run worker(zip, task) for every task, spread over several threads.
    Each thread opens its own connection to the zip, since reads over the
    network are slow one at a time but parallelize well.

    Args:
        zip_name: Filename of the zip in the dataset repo.
        tasks: List of tasks to process.
        worker: Function taking (open ZipFile, task) and returning a result.
        num_workers: Number of threads.
        desc: Label for the progress bar.

    Returns:
        List of worker results (in no particular order).
    """
    if not tasks:
        return []

    num_workers = max(1, min(num_workers, len(tasks)))
    chunks = [tasks[i::num_workers] for i in range(num_workers)]
    progress = tqdm(total=len(tasks), desc=desc)

    def process_chunk(chunk):
        results = []
        with _remote_zip(zip_name) as z:
            for task in chunk:
                results.append(worker(z, task))
                progress.update(1)
        return results

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(process_chunk, chunks))
    progress.close()
    return [result for results in chunk_results for result in results]


def download_instructscene_subset(
    room_types=ROOM_TYPES,
    room_view=DEFAULT_ROOM_VIEW,
    max_rooms_per_type=None,
    include_item_images=True,
    num_workers=8,
    cache_dir=DATA_DIR,
):
    """
    OBS: Takes about 40 minutes to run and 700 MB of storage. 

    Download only what's needed from the InstructScene version of 3D-FRONT:
    one image per room, one image per item, and compact metadata.

    Stored under cache_dir:
      - rooms/{room_type}/{room_uid}.png -> one 256 px render per room.
      - items/{jid}.jpg                 -> one preview image per item.
      - metadata/rooms.json             -> per room: room type, finer
                                           scene type, and its objects
                                           (item jid + position, size,
                                           rotation).
      - metadata/items.json             -> per item: category, style,
                                           theme, material, captions.

    Skipped: the other 8 renders per room, 3D meshes and textures,
    OpenShape/CLIP features, scene graphs, and the pretrained model
    checkpoints in the repo.

    Files are read straight out of the zips on the Hub, so the 5 GB and
    26 GB zips are never downloaded whole. Already-downloaded rooms and
    items are skipped, so an interrupted run can simply be restarted.

    Args:
        room_types: Which room types to download; any of "bedroom",
            "diningroom", "livingroom".
        room_view: Which render to keep per room: "000" to "007"
            (perspective views) or "topdown".
        max_rooms_per_type: If set, only download this many rooms per
            room type (useful for a quick test). None downloads all.
        include_item_images: Whether to download item images. If False,
            only room images and metadata are downloaded.
        num_workers: Number of parallel connections to the Hub.
        cache_dir: Local directory everything is stored in. Defaults to
            data/instructscene/ at the repo root.

    Returns:
        dict with "rooms" and "items" (counts stored locally), and
        "missing_item_images" (jids with no image in the dataset).
    """
    if room_view not in ROOM_VIEWS:
        raise ValueError(f"room_view must be one of {ROOM_VIEWS}, got {room_view!r}")

    rooms, items = _load_local_metadata(cache_dir)

    with _remote_zip(ROOMS_ZIP) as rooms_zip:
        remote_rooms = _list_remote_rooms(rooms_zip, room_types)

    if max_rooms_per_type is not None:
        selected = []
        for room_type in room_types:
            selected += [room for room in remote_rooms if room[0] == room_type][:max_rooms_per_type]
        remote_rooms = selected

    rooms_to_fetch = [
        (room_type, room_uid)
        for room_type, room_uid in remote_rooms
        if room_uid not in rooms or not os.path.exists(_local_room_image_path(cache_dir, room_type, room_uid))
    ]

    def fetch_room(rooms_zip, task):
        room_type, room_uid = task
        try:
            room, room_items = _parse_room(rooms_zip, room_type, room_uid)
            image_bytes = rooms_zip.read(_room_image_in_zip(room_type, room_uid, room_view))
        except (KeyError, ValueError) as error:
            print(f"Skipping room {room_uid}: {error}")
            return None

        image_path = _local_room_image_path(cache_dir, room_type, room_uid)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        room["image_view"] = room_view
        return room, room_items

    for result in _run_on_remote_zip(ROOMS_ZIP, rooms_to_fetch, fetch_room, num_workers, "Rooms"):
        if result is not None:
            room, room_items = result
            rooms[room["room_uid"]] = room
            items.update(room_items)

    # Saved before fetching item images, so room progress isn't lost if
    # the item step is interrupted.
    _save_local_metadata(cache_dir, rooms, items)

    missing_item_images = []
    if include_item_images:
        items_to_fetch = [jid for jid in sorted(items) if not os.path.exists(_local_item_image_path(cache_dir, jid))]

        def fetch_item(items_zip, jid):
            try:
                image_bytes = items_zip.read(_item_image_in_zip(jid))
            except KeyError:
                return jid
            image_path = _local_item_image_path(cache_dir, jid)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            return None

        results = _run_on_remote_zip(ITEMS_ZIP, items_to_fetch, fetch_item, num_workers, "Items")
        missing_item_images = sorted(jid for jid in results if jid is not None)
        if missing_item_images:
            print(f"{len(missing_item_images)} items have no image in {ITEMS_ZIP}")

    return {"rooms": len(rooms), "items": len(items), "missing_item_images": missing_item_images}


def visualize_random_rooms(
    num_rooms=3,
    room_types=ROOM_TYPES,
    room_view=DEFAULT_ROOM_VIEW,
    use_local=True,
    cache_dir=DATA_DIR,
    seed=None,
):
    """
    Sanity-check the data by displaying random rooms next to every item in
    them, with each item's category and style shown above its image.

    Works whether or not the data is downloaded:
      - If use_local is True and rooms have been downloaded to cache_dir,
        rooms are sampled from those and their stored files are used.
        Anything missing locally (e.g. an item image) is read from the Hub.
      - Otherwise rooms are sampled from the whole dataset on the Hub and
        read straight into memory. Nothing is written to disk, so there's
        nothing to clean up afterward.

    This is a visualization/debugging aid for checking that the data
    extraction is correct - it isn't part of the data pipeline itself.

    Args:
        num_rooms: How many random rooms to display.
        room_types: Which room types to sample from.
        room_view: Which render to show for rooms read from the Hub ("000"
            to "007" or "topdown"). Locally stored rooms show the view that
            was downloaded.
        use_local: Whether to use downloaded data when there is some. Set
            False to test the no-download path even with data on disk.
        cache_dir: Local root directory to look for downloaded data in.
            Defaults to data/instructscene/ at the repo root.
        seed: Optional random seed, for reproducible sampling.

    Returns:
        None.
    """
    rng = random.Random(seed)
    local_rooms, local_items = _load_local_metadata(cache_dir) if use_local else ({}, {})

    with ExitStack() as stack:
        open_zips = {}

        def get_zip(zip_name):
            # Each zip is opened at most once, and only if something
            # actually has to be read from it.
            if zip_name not in open_zips:
                open_zips[zip_name] = stack.enter_context(_remote_zip(zip_name))
            return open_zips[zip_name]

        candidates = sorted(
            (room["room_type"], uid) for uid, room in local_rooms.items() if room["room_type"] in room_types
        )
        if not candidates:
            candidates = _list_remote_rooms(get_zip(ROOMS_ZIP), room_types)

        for room_type, room_uid in rng.sample(candidates, k=min(num_rooms, len(candidates))):
            if room_uid in local_rooms:
                room, room_items = local_rooms[room_uid], local_items
            else:
                room, room_items = _parse_room(get_zip(ROOMS_ZIP), room_type, room_uid)

            room_image_path = _local_room_image_path(cache_dir, room_type, room_uid)
            if use_local and os.path.exists(room_image_path):
                room_image = Image.open(room_image_path)
            else:
                room_image = Image.open(io.BytesIO(get_zip(ROOMS_ZIP).read(_room_image_in_zip(room_type, room_uid, room_view))))
            room_image.load()

            images = [room_image]
            labels = [f"{room_type}\n({room['scene_type']})"]
            for obj in room["objects"]:
                jid = obj["jid"]
                item_image_path = _local_item_image_path(cache_dir, jid)
                if use_local and os.path.exists(item_image_path):
                    item_image = Image.open(item_image_path)
                else:
                    try:
                        item_image = Image.open(io.BytesIO(get_zip(ITEMS_ZIP).read(_item_image_in_zip(jid))))
                    except KeyError:
                        continue  # no image for this item in the dataset
                item_image.load()

                info = room_items.get(jid, {})
                images.append(item_image)
                labels.append(f"{info.get('category') or 'unknown category'}\n({info.get('style') or 'unknown style'})")

            max_cols = 7
            num_panels = len(images)
            num_cols = min(max_cols, num_panels)
            num_rows = -(-num_panels // num_cols)  # ceil division

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
            axes = [axes] if num_panels == 1 else axes.flatten()

            for ax, image, label in zip(axes, images, labels):
                ax.imshow(image)
                ax.set_title(label)
                ax.axis("off")

            # Hide any unused panels left over in the last row.
            for ax in axes[num_panels:]:
                ax.axis("off")

            plt.tight_layout()
            plt.show()


def _items_key(room):
    """
    What a room's items boil down to when checking for duplicates: its
    item IDs, sorted, with repeats kept (two identical chairs differ from
    one chair).

    Args:
        room: A room record (as in metadata/rooms.json).

    Returns:
        Tuple of item IDs (jids).
    """
    return tuple(sorted(obj["jid"] for obj in room["objects"]))


def _layout_key(room):
    """
    A room's items together with where they're placed, rounded to 2
    decimals so tiny numeric differences don't count.

    Args:
        room: A room record (as in metadata/rooms.json).

    Returns:
        Tuple, one entry per object, sorted.
    """
    return tuple(
        sorted(
            (
                obj["jid"],
                tuple(round(v, 2) for v in obj["translation"]),
                tuple(round(v, 2) for v in obj["size"]),
                round(obj["angle"], 2),
            )
            for obj in room["objects"]
        )
    )


def _find_duplicate_groups(rooms, min_items):
    """
    Group rooms that contain the same items (same item IDs, same number of
    each). Rooms with fewer than min_items items are ignored.

    Args:
        rooms: Dict of room records keyed by room_uid.
        min_items: Rooms with fewer items than this are ignored.

    Returns:
        Tuple (checked, groups):
          - checked: dict of the rooms that were considered.
          - groups: list of lists of room_uids (each sorted, at least two
            per list), ordered by the first room_uid of each list.
    """
    checked = {uid: room for uid, room in rooms.items() if len(room["objects"]) >= min_items}

    by_items = defaultdict(list)
    for uid in sorted(checked):
        by_items[_items_key(checked[uid])].append(uid)

    groups = [uids for uids in by_items.values() if len(uids) >= 2]
    groups.sort(key=lambda uids: uids[0])
    return checked, groups


def _read_all_rooms_from_hub(room_types, num_workers):
    """
    Read the metadata of every room straight from the Hub into memory
    (nothing is saved). Takes a while, since each room is a separate read.

    Args:
        room_types: Which room types to read.
        num_workers: Number of parallel connections to the Hub.

    Returns:
        Tuple (rooms, items): dicts keyed by room_uid and jid.
    """
    with _remote_zip(ROOMS_ZIP) as rooms_zip:
        remote_rooms = _list_remote_rooms(rooms_zip, room_types)

    def read_room(rooms_zip, task):
        room_type, room_uid = task
        try:
            return _parse_room(rooms_zip, room_type, room_uid)
        except (KeyError, ValueError) as error:
            print(f"Skipping room {room_uid}: {error}")
            return None

    rooms, items = {}, {}
    for result in _run_on_remote_zip(ROOMS_ZIP, remote_rooms, read_room, num_workers, "Reading rooms"):
        if result is not None:
            room, room_items = result
            rooms[room["room_uid"]] = room
            items.update(room_items)
    return rooms, items


def find_and_show_duplicate_rooms(
    num_groups=3,
    rooms_per_group=4,
    min_items=3,
    room_types=ROOM_TYPES,
    room_view=DEFAULT_ROOM_VIEW,
    use_local=True,
    num_workers=8,
    cache_dir=DATA_DIR,
    seed=None,
    only_mixed_room_types=False,
    only_rearranged=False,
):
    """
    Find rooms that contain exactly the same items, display a few of those
    groups, and return how many there are. Nothing is deleted or changed.

    Two rooms count as having the same items if they contain the same
    item IDs, with the same number of each. Rooms with fewer than
    min_items items are ignored, because small rooms (e.g. just one bed)
    often match by coincidence.

    Each displayed group shows the rooms first, then the shared items.
    Whether the rooms in a group also have the same layout (items placed
    in the same positions) is checked and reported: same items AND same
    positions is a stronger sign of a duplicate than same items in
    different positions.

    Room metadata comes from your downloaded data if there is any (and
    use_local is True). Otherwise it's read from the Hub into memory,
    which takes a while because every room has to be read. Images are
    taken from disk when downloaded and read from the Hub otherwise; nothing
    is written to disk.

    Args:
        num_groups: How many duplicate groups to display.
        rooms_per_group: Max rooms to display per group (larger groups
            are cut off, but are counted in full in the return value).
        min_items: Rooms with fewer items than this are ignored.
        room_types: Which room types to check.
        room_view: Which render to show for rooms not downloaded ("000" to
            "007" or "topdown").
        use_local: Whether to use downloaded data when there is some. Set
            False to read everything from the Hub.
        num_workers: Number of parallel connections to the Hub, used only
            when room metadata has to be read from the Hub.
        cache_dir: Local root directory to look for downloaded data in.
            Defaults to data/instructscene/ at the repo root.
        seed: Optional random seed, for reproducible choice of groups.
        only_mixed_room_types: If True, only consider groups whose rooms
            are in different room types (e.g. a bedroom and a livingroom
            with the same items), such as open-plan living/dining rooms.
            Everything below (the counts and the displayed groups) then
            covers only those groups.
        only_rearranged: If True, only consider groups whose rooms don't
            all have the same layout, i.e. the items are placed differently
            (rooms that are mirrored, or have some items moved). Can be
            combined with only_mixed_room_types.

    Returns:
        dict with:
          - "num_rooms_checked": rooms with at least min_items items.
          - "num_groups": groups of rooms with identical items.
          - "num_rooms_in_groups": rooms that are in some group.
          - "num_redundant_rooms": how many rooms would be left over if
            each group were reduced to a single room (rooms in groups
            minus number of groups).
          - "num_groups_same_layout": groups where every room also has
            the same item positions.
          - "groups": list of dicts, one per group, with "room_uids",
            "room_type" (that of the first room), "room_types" (all room
            types in the group, sorted) and "same_layout".
    """
    rooms, items = _load_local_metadata(cache_dir) if use_local else ({}, {})
    if not rooms:
        rooms, items = _read_all_rooms_from_hub(room_types, num_workers)

    rooms = {uid: room for uid, room in rooms.items() if room["room_type"] in room_types}
    checked, uid_groups = _find_duplicate_groups(rooms, min_items)

    groups = []
    for uids in uid_groups:
        group_room_types = sorted({checked[uid]["room_type"] for uid in uids})
        if only_mixed_room_types and len(group_room_types) < 2:
            continue
        layouts = {_layout_key(checked[uid]) for uid in uids}
        if only_rearranged and len(layouts) == 1:
            continue
        groups.append(
            {
                "room_uids": uids,
                "room_type": checked[uids[0]]["room_type"],
                "room_types": group_room_types,
                "same_layout": len(layouts) == 1,
            }
        )

    summary = {
        "num_rooms_checked": len(checked),
        "num_groups": len(groups),
        "num_rooms_in_groups": sum(len(group["room_uids"]) for group in groups),
        "num_redundant_rooms": sum(len(group["room_uids"]) - 1 for group in groups),
        "num_groups_same_layout": sum(group["same_layout"] for group in groups),
        "groups": groups,
    }
    print(
        f"{summary['num_groups']} groups of rooms with identical items"
        f"{' in different room types' if only_mixed_room_types else ''}"
        f"{' and different layouts' if only_rearranged else ''} "
        f"({summary['num_rooms_in_groups']} of {summary['num_rooms_checked']} rooms with at least "
        f"{min_items} items; in {summary['num_groups_same_layout']} of the groups the layouts are identical too)"
    )

    with ExitStack() as stack:
        open_zips = {}

        def get_zip(zip_name):
            # Each zip is opened at most once, and only if something
            # actually has to be read from it.
            if zip_name not in open_zips:
                open_zips[zip_name] = stack.enter_context(_remote_zip(zip_name))
            return open_zips[zip_name]

        def load_room_image(room):
            path = _local_room_image_path(cache_dir, room["room_type"], room["room_uid"])
            if use_local and os.path.exists(path):
                image = Image.open(path)
            else:
                member = _room_image_in_zip(room["room_type"], room["room_uid"], room_view)
                image = Image.open(io.BytesIO(get_zip(ROOMS_ZIP).read(member)))
            image.load()
            return image

        def load_item_image(jid):
            path = _local_item_image_path(cache_dir, jid)
            try:
                if use_local and os.path.exists(path):
                    image = Image.open(path)
                else:
                    image = Image.open(io.BytesIO(get_zip(ITEMS_ZIP).read(_item_image_in_zip(jid))))
            except KeyError:
                return None  # no image for this item in the dataset
            image.load()
            return image

        shown = random.Random(seed).sample(groups, k=min(num_groups, len(groups)))
        for number, group in enumerate(shown, start=1):
            group_rooms = [rooms[uid] for uid in group["room_uids"][:rooms_per_group]]

            images, labels = [], []
            for room in group_rooms:
                images.append(load_room_image(room))
                short_id = room["room_uid"].split("_")[0][:8]
                labels.append(f"{room['room_type']} ({room['scene_type']})\n{short_id}")

            for jid in sorted(set(_items_key(group_rooms[0]))):
                image = load_item_image(jid)
                if image is None:
                    continue
                info = items.get(jid, {})
                images.append(image)
                labels.append(f"{info.get('category') or 'unknown category'}\n({info.get('style') or 'unknown style'})")

            max_cols = 7
            num_panels = len(images)
            num_cols = min(max_cols, num_panels)
            num_rows = -(-num_panels // num_cols)  # ceil division

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
            axes = [axes] if num_panels == 1 else axes.flatten()

            for ax, image, label in zip(axes, images, labels):
                ax.imshow(image)
                ax.set_title(label)
                ax.axis("off")
            for ax in axes[num_panels:]:
                ax.axis("off")

            layout = "same layout" if group["same_layout"] else "different layouts"
            cut_off = f", showing {len(group_rooms)}" if len(group["room_uids"]) > len(group_rooms) else ""
            fig.suptitle(
                f"Group {number}: {len(group['room_uids'])} rooms with identical items "
                f"({', '.join(group['room_types'])}; {layout}{cut_off}). "
                f"Rooms first, then the shared items.",
                fontsize=14,
            )
            plt.tight_layout()
            plt.show()

    return summary



def delete_duplicate_rooms(
    min_items=3,
    room_types=ROOM_TYPES,
    use_local=True,
    num_workers=8,
    dry_run=True,
    cache_dir=DATA_DIR,
):
    """
    Remove duplicate rooms, keeping exactly one room per group of rooms with
    the same items, and make sure the removed rooms never come back.

    Duplicates are rooms with the same item IDs (same number of each),
    whatever their layout, so mirrored rooms and rooms with small position
    differences count as duplicates. A different scene_type (e.g. "bedroom"
    vs "masterbedroom") is ignored. Rooms with fewer than min_items items are
    ignored, as in find_and_show_duplicate_rooms.

    Duplicates are removed within a room type. A group with rooms in several
    room types (typically open-plan living/dining rooms, which exist as a
    livingroom and a diningroom) keeps one room in each of those types, so
    the room stays in both classes, and only extra copies within the same
    type are removed.

    The room kept is the first one when sorted by room_uid. That is
    arbitrary but deterministic, so it's the same on every machine.

    With dry_run=True (the default) only prints and returns what would
    happen. Otherwise:
      1. The removed room_uids are added to EXCLUDED_ROOMS_PATH
         (src/data_generation/instructscene_excluded_rooms.txt). Commit this
         file so everyone ignores the same rooms. The download and
         visualization functions skip every room in it, whether the data is
         downloaded or read from the Hub.
      2. Downloaded images of the removed rooms are deleted and the rooms
         are dropped from metadata/rooms.json. Item images stay, since other
         rooms may use them.

    Which rooms are duplicates is decided from the whole dataset: downloaded
    metadata is only used if it covers every room of room_types, otherwise
    all rooms are read from the Hub into memory (slow, nothing is saved).
    This keeps the result the same no matter how much you have downloaded.

    Args:
        min_items: Rooms with fewer items than this are ignored.
        room_types: Which room types to deduplicate.
        use_local: Whether to use downloaded metadata when it covers all
            rooms. Set False to always read from the Hub.
        num_workers: Number of parallel connections to the Hub, used only
            when room metadata has to be read from the Hub.
        dry_run: If True, change nothing and only report.
        cache_dir: Local root directory where downloaded data is stored.
            Defaults to data/instructscene/ at the repo root.

    Returns:
        dict with:
          - "dry_run": whether anything was changed.
          - "num_groups": sets of duplicate rooms within one room type
            (a group spanning two room types counts once per type that has
            more than one room in it).
          - "num_removed": rooms removed (or that would be, in a dry run).
          - "removed_per_room_type": dict room_type -> rooms removed.
          - "groups": list of dicts with "keep", "remove" (list of
            room_uids) and "room_type".
          - "mixed_type_groups": groups (lists of room_uids) whose rooms
            span several room types. One room per type is kept in each.
          - "num_groups_different_layout": groups in "groups" whose rooms
            don't all have identical layouts (includes mirrored rooms).
    """
    with _remote_zip(ROOMS_ZIP) as rooms_zip:
        remote_uids = {uid for _, uid in _list_remote_rooms(rooms_zip, room_types)}

    local_rooms, _ = _load_local_metadata(cache_dir) if use_local else ({}, {})
    local_rooms = {uid: room for uid, room in local_rooms.items() if room["room_type"] in room_types}
    if use_local and remote_uids and remote_uids <= set(local_rooms):
        rooms = local_rooms
    else:
        if local_rooms:
            print("Downloaded metadata doesn't cover every room, so reading all rooms from the Hub instead.")
        rooms, _ = _read_all_rooms_from_hub(room_types, num_workers)
    rooms = {uid: room for uid, room in rooms.items() if room["room_type"] in room_types}

    checked, uid_groups = _find_duplicate_groups(rooms, min_items)

    groups, mixed_type_groups = [], []
    for uids in uid_groups:
        uids_per_type = defaultdict(list)
        for uid in uids:
            uids_per_type[checked[uid]["room_type"]].append(uid)
        if len(uids_per_type) > 1:
            mixed_type_groups.append(uids)

        # Duplicates are only removed within a room type.
        for room_type, type_uids in sorted(uids_per_type.items()):
            if len(type_uids) < 2:
                continue
            groups.append(
                {
                    "keep": type_uids[0],
                    "remove": type_uids[1:],
                    "room_type": room_type,
                    "different_layout": len({_layout_key(checked[uid]) for uid in type_uids}) > 1,
                }
            )

    removed_per_room_type = defaultdict(int)
    for group in groups:
        removed_per_room_type[group["room_type"]] += len(group["remove"])
    to_remove = [uid for group in groups for uid in group["remove"]]

    summary = {
        "dry_run": dry_run,
        "num_groups": len(groups),
        "num_removed": len(to_remove),
        "removed_per_room_type": dict(removed_per_room_type),
        "groups": [{key: group[key] for key in ("keep", "remove", "room_type")} for group in groups],
        "mixed_type_groups": mixed_type_groups,
        "num_groups_different_layout": sum(group["different_layout"] for group in groups),
    }

    verb = "Would remove" if dry_run else "Removing"
    print(
        f"{verb} {len(to_remove)} of {len(checked)} rooms with at least {min_items} items "
        f"({len(groups)} groups, one room kept per group and room type): {summary['removed_per_room_type']}"
    )
    print(
        f"{summary['num_groups_different_layout']} of the groups have rooms with different layouts "
        f"(mirrored or moved items); the rest are identical."
    )
    if mixed_type_groups:
        print(
            f"{len(mixed_type_groups)} groups span several room types; "
            f"one room is kept in each type, only extra copies within a type are removed."
        )

    if dry_run or not to_remove:
        return summary

    # The exclusion list is written first, so the rooms stay out even if
    # something fails while deleting the local files.
    _save_excluded_rooms(_load_excluded_rooms() | set(to_remove))

    for uid in to_remove:
        room_type = rooms[uid]["room_type"]
        image_path = _local_room_image_path(cache_dir, room_type, uid)
        if os.path.exists(image_path):
            os.remove(image_path)

    all_local_rooms, items = _load_local_metadata(cache_dir)
    if all_local_rooms:
        _save_local_metadata(cache_dir, {uid: r for uid, r in all_local_rooms.items() if uid not in set(to_remove)}, items)

    print(f"Added {len(to_remove)} rooms to {os.path.basename(EXCLUDED_ROOMS_PATH)}. Remember to commit it.")
    return summary


if __name__ == "__main__":
    visualize_random_rooms(room_types="diningroom")
    #find_and_show_duplicate_rooms(num_groups=10)
    #delete_duplicate_rooms(dry_run=False)