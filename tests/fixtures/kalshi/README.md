# Kalshi REST fixture provenance

These minimized payloads mirror the official Trade API v2 response schemas checked on
2026-09-03. They are deterministic offline fixtures, not claims about a currently listed live
market.

- Markets: https://docs.kalshi.com/api-reference/market/get-markets
- Events: https://docs.kalshi.com/api-reference/events/get-event
- Event fee changes: https://docs.kalshi.com/api-reference/events/get-event-fee-changes
- Series: https://docs.kalshi.com/api-reference/market/get-series
- Order books: https://docs.kalshi.com/getting_started/orderbook_responses
- Fixed point: https://docs.kalshi.com/getting_started/fixed_point_migration
- Prediction-contract fee schedule (effective 2026-07-07):
  https://kalshi.com/docs/kalshi-fee-schedule.pdf
- Fee rounding and account precision:
  https://docs.kalshi.com/getting_started/fee_rounding

`fees_2026-07-07.json` records the source URLs, schedule effective date, capture date, policy
version, official taker table values, and official non-direct/FCM rounding example.
`event_fee_changes.json` preserves paired future-override and clear examples plus one unknown
additive field to verify tolerant wire parsing and raw-payload retention.
