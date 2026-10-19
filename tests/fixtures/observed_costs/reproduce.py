"""Manually reproduce licensed observations; never invoked by the test suite.

Usage: python reproduce.py RAW_CACHE --output OUTPUT
Missing pinned shards are downloaded into RAW_CACHE. No venue credentials.
"""
from __future__ import annotations

import argparse
import bisect
import calendar
import hashlib
import json
import time
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import polars as pl

REVISION = '54b07b4bebc5574a065d3fbf4432634cf9148f50'
DATES = ('2026-06-01', '2026-06-29', '2026-08-01')
PINS = {
    ('markets', DATES[0]): ('5c1fb4d5a5a7d1bc83f25365dd5849e7b29a7b55c6c7fc1e6c2a8f7e40d238d2', '2026-06-04T02:37:21Z'),
    ('markets', DATES[1]): ('0a65f9ca98e46ae6f22fbf92b26c87048fe78ee93c49192e2110e3f37f31cf80', '2026-07-02T02:33:33Z'),
    ('markets', DATES[2]): ('1bdd00b3815a2d49def09b91e48bec103149e339fd10329dfe7f1db6625ca6a0', '2026-08-04T02:33:11Z'),
    ('trades', DATES[0]): ('f25503abfb3bea4a54cc79b150e600db62f8dc962aaf375788a31fcdc5b27908', '2026-06-02T02:20:25Z'),
    ('trades', DATES[1]): ('1c7b6c31d28c36f56c4e8990bad2c4987a79f2845c16867e3a6f424959ebb4c3', '2026-06-30T02:20:25Z'),
    ('trades', DATES[2]): ('a039f78d45f6b53d6039689aa5de67728f5656209aa1526a7ca4eeeb724e7210', '2026-08-02T02:20:24Z'),
}
BOUNDARIES = {
    DATES[0]: '485709d4f038ef8afdce3cffb46248a8642489d3100a77031880248d6ffa36f5',
    DATES[1]: '1c3e65333bb6d9916cfbaeff97474b33c32bf5c236a8491f8ffbac13517dbaaf',
    DATES[2]: 'b6923391717d4982cd2c8c856b97f5ab59e8ac2c4638096b7e1d55ce9b524885',
}
REFERENCE = 'chainticks:hyperliquid:BTC:minute_midpoint_label:' + REVISION
LICENSE = {
    'license': 'CC BY 4.0', 'url': 'https://creativecommons.org/licenses/by/4.0/',
    'source': 'https://huggingface.co/datasets/Chainticks/perp-data', 'revision': REVISION,
    'attribution': 'Chainticks Perp Data, derived from public chain/archive state. https://chainticks.com',
    'modified': 'normalized/hashed; compact selected population and separately labelled diagnostics',
    'identity_notice': 'Deterministic account hashes normalize identity; they are not anonymization.',
    'publisher_license_text': 'Chainticks Dataset License\n\nCreative Commons Attribution 4.0 International (CC BY 4.0)\n\nYou are free to share and adapt this dataset, including for commercial purposes, provided that appropriate credit is given.\n\nCanonical license text: https://creativecommons.org/licenses/by/4.0/\n\nRecommended attribution:\n\n"Chainticks Perp Data, derived from public chain/archive state. https://chainticks.com"\n\nChainticks asserts no ownership over underlying market facts. Chainticks is independent and is not affiliated with, endorsed by, or sponsored by Hyperliquid Labs or any protocol whose data appears in this dataset.\n',
    'publisher_notice': 'Chainticks asserts no ownership over underlying market facts and is independent of Hyperliquid Labs.',
}


def ns(stamp):
    base, dot, fraction = stamp.removesuffix('Z').partition('.')
    seconds = calendar.timegm(datetime.strptime(base, '%Y-%m-%dT%H:%M:%S').timetuple())
    return seconds * 1_000_000_000 + (int(fraction.ljust(9, '0')) if dot else 0)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def source_file(cache, kind, day, part=0):
    relative = f'hyperliquid_chain/{kind}/date={day}/part-{part:04}.parquet'
    uri = f'https://huggingface.co/datasets/Chainticks/perp-data/resolve/{REVISION}/{relative}'
    expected = PINS[kind, day][0] if part == 0 else BOUNDARIES[day]
    path = cache / f'{kind}-{day}-part{part}.parquet'
    if not path.exists():
        urllib.request.urlretrieve(uri, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Source hash mismatch: {relative}')
    return path, uri


def normalize(raw, ordinal, day, mids, times):
    event, block = raw['event'], raw['block']
    if not isinstance(event['user'], str) or event['tid'] is None:
        raise ValueError('Missing account/trade identity')
    account = hashlib.sha256(('hyperliquid:mainnet:' + event['user']).encode()).hexdigest()
    leg = hashlib.sha256(f"{account}:{event['oid']}:{event['tid']}".encode()).hexdigest()
    time_ns = event['time'] * 1_000_000
    index = bisect.bisect_right(times, time_ns) - 1
    reference = mids[index] if index >= 0 else None
    return {
        'leg_id': leg, 'account': account, 'parent_id': None if event['oid'] is None else str(event['oid']),
        'source_asset_sha256': PINS['trades', day][0], 'source_row': ordinal,
        'block_number': block['block_number'], 'intra_block_index': None,
        'block_time_ns': ns(block['block_time']),
        'node_local_time_ns': ns(block['local_time']) if block.get('local_time') else None,
        'side': 'buy' if event['side'] == 'B' else 'sell',
        'role': 'taker' if event['crossed'] is True else 'maker' if event['crossed'] is False else 'unknown',
        'mechanism': None, 'quantity': event['sz'],
        'quote_notional': str(Decimal(event['sz']) * Decimal(event['px'])),
        'time_ns': time_ns, 'received_at_ns': ns(PINS['trades', day][1]),
        'fee': event.get('fee'), 'fee_currency': event.get('feeToken'),
        'builder_fee': event.get('builderFee'),
        'twap_id': None if event.get('twapId') is None else str(event['twapId']),
        'fee_received_at_ns': ns(PINS['trades', day][1]) if event.get('fee') is not None else None,
        'reference_price': None if reference is None else reference['mid_px'],
        'reference_time_ns': None if index < 0 else times[index],
        'reference_received_at_ns': ns(PINS['markets', day][1]) if reference is not None else None,
    }


def estimates(rows):
    result = {}
    for side, sign in [('buy', 1), ('sell', -1)]:
        group = [row for row in rows if row['side'] == side]
        reference = sum(Decimal(row['quantity']) * Decimal(row['reference_price']) for row in group)
        quote = sum(Decimal(row['quote_notional']) for row in group)
        price = sum(sign * (Decimal(row['quote_notional']) - Decimal(row['quantity']) * Decimal(row['reference_price'])) for row in group)
        result[side] = {'price_cost_bps': str(10_000 * price / reference),
                        'fee_bps': str(10_000 * sum(Decimal(row['fee']) for row in group) / quote),
                        'legs': len(group), 'parents': len({(row['account'], row['parent_id']) for row in group})}
    return result


def reproduce(cache, output, capture_ns=None):
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    assets, admitted, diagnostic, rebates, benchmark, boundary = [], [], [], [], [], {}
    summary, sensitivity, ordinal_checks = {}, {}, {}
    existing = output / 'provenance.json'
    captured = capture_ns if capture_ns is not None else (json.loads(existing.read_text())['normalized_at_ns'] if existing.exists() else time.time_ns())
    for day in DATES:
        start = ns(day + 'T00:00:00Z')
        for kind in ('markets', 'trades'):
            _, uri = source_file(cache, kind, day)
            assets.append({'locator': uri, 'sha256': PINS[kind, day][0],
                           'receipt_bound_ns': ns(PINS[kind, day][1]),
                           'receipt_basis': 'publisher_commit', 'kind': 'benchmark' if kind == 'markets' else 'fills'})
        market_path, _ = source_file(cache, 'markets', day)
        market = pl.read_parquet(market_path).with_row_index('__source_row').filter(pl.col('symbol') == 'BTC')
        mids = [json.loads(value) for value in market['raw_json']]
        times = [ns(row['time']) for row in mids]
        if times != sorted(times) or {start + i * 60_000_000_000 for i in range(20)} - set(times):
            raise ValueError('Missing or unordered minute benchmarks')
        trade_path, _ = source_file(cache, 'trades', day)
        trade = pl.read_parquet(trade_path).with_row_index('__source_row').filter(pl.col('symbol') == 'BTC')
        observed = [(int(index), json.loads(value)) for index, value in zip(trade['__source_row'], trade['raw_json'], strict=True)]
        if not start <= observed[0][1]['event']['time'] * 1_000_000 < start + 60_000_000_000 or observed[-1][1]['event']['time'] * 1_000_000 <= start + 1_200_000_000_000:
            raise ValueError('First trade shard does not cover selected window')
        following, uri = source_file(cache, 'trades', day, 1)
        first = json.loads(pl.read_parquet(following, n_rows=1, columns=['raw_json'])['raw_json'][0])
        if first['event']['time'] * 1_000_000 < start + 1_200_000_000_000:
            raise ValueError('Selected window crosses a shard boundary')
        boundary[day] = {'locator': uri, 'sha256': BOUNDARIES[day], 'first_time_ms': first['event']['time']}
        counts = Counter()
        day_sides = defaultdict(Counter)
        notionals = defaultdict(Decimal)
        wider = []
        rebate = None
        for ordinal, raw in observed:
            event = raw['event']
            time_ns = event['time'] * 1_000_000
            if not start <= time_ns < start + 1_200_000_000_000:
                continue
            fill = normalize(raw, ordinal, day, mids, times)
            counts[fill['role'] + '_legs'] += 1
            side = day_sides[fill['side']]
            side[fill['role'] + '_legs'] += 1
            notionals[fill['side'] + ':' + fill['role']] += Decimal(fill['quote_notional'])
            if fill['twap_id'] is not None:
                notionals[fill['side'] + ':' + fill['role'] + ':twap'] += Decimal(fill['quote_notional'])
            if fill['role'] == 'maker' and Decimal(fill['fee']) < 0 and rebate is None:
                rebate = fill
            if fill['role'] != 'taker':
                continue
            counts['taker_twap_legs'] += int(fill['twap_id'] is not None)
            age = time_ns - fill['reference_time_ns']
            if age <= 4_000_000_000:
                wider.append(fill)
            if time_ns < start + 60_000_000_000:
                diagnostic.append(fill)
            if age <= 1_000_000_000:
                admitted.append(fill)
                counts['selected_legs'] += 1
                counts['selected_twap_legs'] += int(fill['twap_id'] is not None)
                side['selected_legs'] += 1
            else:
                counts['age_excluded_legs'] += 1
                side['age_excluded_legs'] += 1
        if rebate is None:
            raise ValueError('Expected an actual maker rebate witness')
        rebates.append(rebate)
        summary[day] = {**dict(counts), 'day_side': {name: dict(value) for name, value in day_sides.items()},
                        'quote_notionals': {name: str(value) for name, value in sorted(notionals.items())}}
        sensitivity[day] = estimates(wider)
        ordinal_checks[day] = {'market_btc_rows': market.height, 'trade_btc_rows': trade.height}
        if day == DATES[2]:
            for time_ns, row in zip(times, mids, strict=True):
                if any(start + hour * 3_600_000_000_000 <= time_ns <= start + hour * 3_600_000_000_000 + 420_000_000_000 for hour in (1, 2, 3)):
                    benchmark.append({'row_id': f'mid:{time_ns}', 'phase': 'point', 'time_ns': time_ns,
                                      'replay_available_at_ns': time_ns, 'price': float(Decimal(row['mid_px'])),
                                      'reference_id': REFERENCE, 'observed_receipt_ns': ns(PINS['markets', day][1])})
    source = {'identity': 'Chainticks/perp-data@' + REVISION, 'assets': assets,
              'venue': 'hyperliquid', 'environment': 'mainnet',
              'product': {'kind': 'linear_perpetual', 'instrument': 'BTC', 'base_currency': 'BTC', 'quote_currency': 'USDC'},
              'captured_at_ns': captured, 'reference_id': REFERENCE, 'reference_convention': 'minute_midpoint_label',
              'windows': [[ns(day+'T00:00:00Z'), ns(day+'T00:20:00Z')] for day in DATES], 'supersedes': None}
    diagnostics = {'qualification': 'post_hoc_not_preregistered', 'method_age_ns': 1_000_000_000,
                   'wider_age_4s_daily': sensitivity, 'one_minute_all_age': estimates(diagnostic),
                   'archive_latest_date': '2026-08-11', 'archive_last_modified': '2026-08-12',
                   'reference_time': 'minute_label_snapshot_inferred', 'fee_tier': 'unknown_account_specific'}
    members = {}
    for name, rows in [('admitted.json', admitted), ('sensitivity-one-minute.json', diagnostic), ('maker-rebates.json', rebates)]:
        rows.sort(key=lambda row: (row['time_ns'], row['block_number'], row['source_asset_sha256'], row['source_row'], row['leg_id']))
        selected_count = sum(row['role'] == 'taker' and row['time_ns'] - row['reference_time_ns'] <= 1_000_000_000 for row in rows)
        members[name] = encode({'version': 1, 'source': source, 'fills': rows,
                                'source_window_summary': {'selected_legs': selected_count, 'days': summary},
                                'source_diagnostics': diagnostics})
    with zipfile.ZipFile(output/'history.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, members[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    provenance = {'revision': REVISION, 'normalized_at_ns': captured, 'source_assets': assets, 'boundary_evidence': boundary,
                  'coverage': ordinal_checks, 'goldens': estimates(admitted), 'diagnostics': diagnostics,
                  'members': {name: {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)} for name, value in members.items()},
                  'archive_sha256': hashlib.sha256((output/'history.zip').read_bytes()).hexdigest(),
                  'zip_compression': {'method': 'deflate', 'level': 9, 'zlib': zlib.ZLIB_VERSION},
                  'benchmark_rows': len(benchmark), 'license': LICENSE,
                  'mainnet_crosscheck': {'tid': 849494584273138, 'oid': 450567938381, 'time_ms': 1780272000533,
                                        'quantity': '0.001', 'price': '73658.0', 'fee': '0.033146', 'fee_currency': 'USDC',
                                        'evidence_sha256': 'c7f1432088a6b6c2d51e457d44750afebc95ca93b656129cf7265cb5e1f35241',
                                        'scope': 'one independently matched public mainnet fill; not an archive-wide audit'}}
    for name, value in [('benchmark.json', benchmark), ('provenance.json', provenance), ('license.json', LICENSE)]:
        (output/name).write_bytes(encode(value))
    total = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    if total > 1024*1024 or len(admitted) != 363 or len(diagnostic) != 1262 or len(benchmark) != 24:
        raise ValueError('Pinned fixture population or byte budget differs')
    print(json.dumps({'selected_legs': len(admitted), 'diagnostic_legs': len(diagnostic), 'benchmark_rows': len(benchmark), 'fixture_bytes': total, 'goldens': provenance['goldens']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_cache', type=Path)
    parser.add_argument('--capture-ns', type=int, help='Preserve the recorded original normalization instant for byte reproduction')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    arguments = parser.parse_args()
    reproduce(arguments.raw_cache, arguments.output, arguments.capture_ns)
