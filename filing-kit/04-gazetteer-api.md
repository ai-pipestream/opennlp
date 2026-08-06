# gazetteer-api

## JIRA fields

*Type:* New Feature. *Components:* opennlp-api, extensions (new module opennlp-geo). *Note:* bundles a project-authored public-domain data table; LICENSE section included; licensing posture per LEGAL-732.

## JIRA description (wiki markup)

```
h2. Summary

OpenNLP can find LOCATION entities but cannot say where they are: there is no gazetteer lookup and no geocoding. This ticket adds the location-intelligence seam and its first implementation. The contracts land in opennlp-api (package {{opennlp.tools.geo}}): {{GeoPoint}}, {{GazetteerEntry}}, {{AttributeValue}}, {{Gazetteer}}, {{GeoResolution}}, and {{Geocoder}}. The shape is generic by design: neutral field names, source-scoped record ids, and a provenance-tagged attribute map for everything dataset-specific, so the dataset is an implementation and never the contract. A documented external-id key convention ({{fips}}, {{geoid}}, {{zcta}}, {{wikidata}}, {{geonames}}, {{whosonfirst}}) lets downstream consumers join census, tax, and municipal open data through stable keys without OpenNLP ever bundling those tables, and the ISO 3166-1 alpha-2 region lookup is the join key shared with the OPENNLP-1870 emoji annotation layer. Resolution results attach to Name Finder output (entity spans in original coordinates), not to the per-token normalization stack.

The first implementation ships in a new {{opennlp-extensions/opennlp-geo}} module: {{BundledGazetteer}}, reading a project-authored 7,342-row public-domain table derived from Natural Earth Populated Places, and {{PopulationPriorGeocoder}}, a deterministic population-plus-feature-class ranker. The licensing posture follows the LEGAL-732 answers: public-domain data is Category A with a LICENSE section and no NOTICE entry; richer tiers (Overture as Category A bundled, GeoNames as user-download, OSM as bring-your-own-data) sit behind the same interface later without a contract change.

h2. Scope

* Contracts in opennlp-api: {{GeoPoint}} (range-validated WGS84), {{GazetteerEntry}} (neutral fields plus the provenance-tagged attribute map), {{Gazetteer}} ({{lookup}}, {{byId}}, {{byRegion}}, {{sources}}), {{GeoResolution}} (mention span in original document coordinates, entry, confidence), {{Geocoder}} (document-level {{resolve(text, mentions)}}, so a context-disambiguating implementation is a drop-in later).
* {{BundledGazetteer}}: fail-loud cursor parser for the bundled table, lazy single load, name matching through the shared OPENNLP-1850 normalization chain.
* {{PopulationPriorGeocoder}}: deterministic ranking by population then feature-class prior, documented heuristic confidence, unresolved mentions omitted.
* The bundled table: derived from Natural Earth 1:10m Populated Places, NAMEASCII canonical, carrying the published wikidata, geonames, and whosonfirst ids (no fabricated identifiers). Audit tests enforce the data file's documented claims row by row.
* LICENSE gains the Natural Earth public-domain section.

h2. Acceptance criteria

* The contracts compile against no new dependencies and reference no dataset by name.
* Audit tests verify the bundled table row by row against its documented claims (row count, id coverage, coordinate ranges).
* The api and opennlp-geo suites are green and the full reactor verify passes.
* The data derivation is documented in the PR (source, upstream commit, canonicalization choices).

h2. Out of scope

* opennlp-distr wiring and the binary-distribution LICENSE section (follow-up).
* A context-minimization geocoder (a second {{Geocoder}} implementation; the document-level contract is designed for it).
* The Overture bundled tier and the GeoNames downloader tool (phase 2, licensing posture already settled).
* The gRPC mirror of {{GeoResolution}} (after the library shape survives one review round).
* Containment-hierarchy and place-similarity features (later phases on the same seam).
```

## PR title

```
<KEY>: Gazetteer and geocoder API with a bundled Natural Earth implementation
```

## PR body (markdown)

```
Adds the location-intelligence seam and its first implementation.

**Contracts** in opennlp-api (`opennlp.tools.geo`): `GeoPoint`, `GazetteerEntry`, `AttributeValue`, `Gazetteer`, `GeoResolution`, `Geocoder`. Generic in shape: the dataset is an implementation, never the contract. Attributes are provenance-tagged, with a documented external-id key convention (`fips`, `geoid`, `zcta`, `wikidata`, `geonames`, `whosonfirst`) so downstream enrichment joins on stable keys, and ISO 3166-1 alpha-2 is the region join key shared with the emoji annotation layer. No new dependencies.

**Implementation** in a new `opennlp-extensions/opennlp-geo` module:

- `BundledGazetteer` over a project-authored 7,342-row public-domain table derived from Natural Earth Populated Places: fail-loud cursor parser, lazy single load, matching through the shared NFC/case/accent folding chain.
- `PopulationPriorGeocoder`: deterministic population-plus-feature-class ranking, documented heuristic confidence, unresolved mentions omitted.

LICENSE gains the Natural Earth public-domain section; audit tests enforce the data file's documented claims row by row.

**Data derivation:** Natural Earth 1:10m Populated Places via the nvkelso GitHub mirror (upstream commit 789c9904, VERSION 5.2.0-pre), NAMEASCII canonical, wikidata/geonames/whosonfirst ids carried as published, no fabricated fips/geoid/zcta.

Verification: api 328/0 (42 new), opennlp-geo 55/0, 19-module reactor verify green.

Follow-ups (deliberately out of this PR): opennlp-distr wiring plus the binary LICENSE section, a context-minimization `Geocoder`, the Overture and GeoNames tiers, and the gRPC mirror after one review round.
```
