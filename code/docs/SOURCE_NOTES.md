# Data sources and credit

This release contains preparation and evaluation code, prepared EPFL data and
a separate CU-BEMS data asset. BPA and ACN records are obtained from their
providers. Model weights and prediction caches are not bundled. Original code
uses MIT; each source collection retains its own license and notices.

## BPA CTA-2045 water heaters

Provider: Bonneville Power Administration. Obtain `winter2018.zip`,
`spring2018.zip` and `summer2018.zip` from the
[BPA project page](https://www.bpa.gov/energy-and-services/conservation/emerging-technologies/portfolio/cta-2045-water-heaters).
The original archive filenames identify the seasonal 2018 release. The
[BPA quickstart](BPA_QUICKSTART.md) reads these existing archives locally.

Credit Bonneville Power Administration (2018), *CTA-2045 Water Heater
Demonstration Report Including A Business Case for CTA-2045 Market
Transformation*, Technology Innovation Project 336, November 9, 2018, and the
data landing page. No dataset-specific redistribution license was established
in the inspected provider materials. Do not treat public download access as a
grant to redistribute raw or derived records.

## ACN workplace charging

The original dataset is ACN-Data, developed with PowerFlex Systems. This runtime
uses the [author-linked static release](https://github.com/tongxin-li/ACN-Data-Static/tree/abb7cf15cc7108913e965d4375cc7270d532b65f)
at revision `abb7cf15cc7108913e965d4375cc7270d532b65f`. The supported sites are
Caltech and office001 (the latter maps to the original `office_01` folder).
See the [ACN quickstart](ACN_QUICKSTART.md) for the exact calendar selection,
planning-only default, bounded acquisition sample and explicit full download.

Cite Zachary J. Lee, Tongxin Li and Steven H. Low (2019), *ACN-Data: Analysis
and Applications of an Open EV Charging Dataset*, e-Energy '19,
[doi:10.1145/3307772.3328313](https://doi.org/10.1145/3307772.3328313), and identify
the static release revision. Its README states that it has no standalone
license and directs readers to [official ACN access information](https://ev.caltech.edu/dataset)
before redistribution. This runtime grants no source-data redistribution
permission. The optional API declaration track is not part of this static
source preparation, and no API credential is required by this runtime.

## EPFL level-3 charging

Provider: Distributed Electrical Systems Laboratory (DESL), EPFL.
Credit *Level 3 electric vehicle charging dataset*,
[repository revision 680bc3bb751f86a47db32a5641f09dcfa079caae](https://github.com/DESL-EPFL/Level-3-EV-charging-dataset/tree/680bc3bb751f86a47db32a5641f09dcfa079caae).
The [EPFL quickstart](EPFL_QUICKSTART.md) downloads the two original workbooks
with their README and MIT LICENSE, retaining the original copyright and
permission text. Keep that original notice with any source copies. It is the
provider's repository notice, not the license for our benchmark code.

The records cover two plugs; EPFL supports device evaluation, not a fleet
group task. The runtime retains the original timestamps because external
timezone alignment has not been established.

## CU-BEMS air conditioning

Provider: Chulalongkorn University Building Energy Management System. Use the
[publisher's version 6 release](https://doi.org/10.6084/m9.figshare.11726517.v6),
which identifies CC BY 4.0. Credit M. Pipattanasomporn, G. Chitalia, J. Songsiri,
C. Aswakul, W. Pora, S. Suwankawin, K. Audomvongseree and N. Hoonchareon, and
retain the source DOI and license. The [CU-BEMS quickstart](CU_BEMS_QUICKSTART.md)
starts from an already acquired original 2019 floor CSV.

The runtime selects AC power channels, retains zone sensor units/masks, forms
history/future windows and defines chronological splits and complete floor/zone
groups. These are derived forecasting-task transformations. They do not modify
the original measurements or add future control labels. The seven floors are
one Bangkok building. Prepared inputs, targets, masks and chronological split
metadata for all seven floors are supplied in the separate data asset.

## Release status

Provider notices were checked for the October 2026 public package. The release
includes code, the manuscript, aggregate results and the licensed prepared
data described above. Source guides record earlier author-run reconstruction
checks; see the root reproduction guide for the current release's entry points.
