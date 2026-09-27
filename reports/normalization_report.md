# Normalization Report (Phase 2)

Generated: 2026-09-26 10:42:07

## Built-in example transformations

### Input: name='Acme Corp.', address='123 Main St, Suite 400, 94105', country='USA'

- name.exact_normalized = `acme corp`
- name.suffix_stripped = `acme`
- name.alnum_normalized = `acmecorp`
- name.token_sorted = `acme corp`
- address.address_normalized = `123 main street suite 400 94105`
- address.house_number = `123`
- address.postal_code = `94105`
- address.unit_token = `400`
- address.street_tokens = `['main', 'street']`
- country.normalized = `united states`

### Input: name='ACME  CORPORATION', address='123 Main Street Ste 400 94105', country='U.S.A'

- name.exact_normalized = `acme corporation`
- name.suffix_stripped = `acme`
- name.alnum_normalized = `acmecorporation`
- name.token_sorted = `acme corporation`
- address.address_normalized = `123 main street ste 400 94105`
- address.house_number = `123`
- address.postal_code = `94105`
- address.unit_token = `400`
- address.street_tokens = `['main', 'street']`
- country.normalized = `united states`

### Input: name='Sharma & Sons Pvt. Ltd.', address='Flat 12B, MG Road, 560001', country='India'

- name.exact_normalized = `sharma and sons pvt ltd`
- name.suffix_stripped = `sharma and sons`
- name.alnum_normalized = `sharmaandsonspvtltd`
- name.token_sorted = `and ltd pvt sharma sons`
- address.address_normalized = `flat 12b mg road 560001`
- address.house_number = `None`
- address.postal_code = `560001`
- address.unit_token = `None`
- address.street_tokens = `['flat', '12b', 'mg', 'road']`
- country.normalized = `india`

### Input: name='Boulangerie Dupont', address='45 Rue de la Paix, 75002', country='France'

- name.exact_normalized = `boulangerie dupont`
- name.suffix_stripped = `boulangerie dupont`
- name.alnum_normalized = `boulangeriedupont`
- name.token_sorted = `boulangerie dupont`
- address.address_normalized = `45 rue de la paix 75002`
- address.house_number = `45`
- address.postal_code = `75002`
- address.unit_token = `None`
- address.street_tokens = `['rue', 'de', 'la', 'paix']`
- country.normalized = `france`

### Input: name="O'Brien's Pub", address="10 O'Connell St., Dublin 1", country='Ireland'

- name.exact_normalized = `o brien s pub`
- name.suffix_stripped = `o brien s pub`
- name.alnum_normalized = `obrienspub`
- name.token_sorted = `brien o pub s`
- address.address_normalized = `10 o connell street dublin 1`
- address.house_number = `10`
- address.postal_code = `None`
- address.unit_token = `None`
- address.street_tokens = `['o', 'connell', 'street', 'dublin', '1']`
- country.normalized = `ireland`

## Sample from train_source1.tsv (first 10 rows)

- raw_name="Orelee's Barbershop" -> exact=`orelee s barbershop` suffix_stripped=`orelee s barbershop`
  raw_address='1795 Westchester Drive, High Point, NC' -> normalized=`1795 westchester drive high point nc` house_number=`1795` postal_code=`None`
  raw_country='US' -> normalized=`united states`

- raw_name='Prime Money' -> exact=`prime money` suffix_stripped=`prime money`
  raw_address='17560 Ellis Road, Tahlequah, OK' -> normalized=`17560 ellis road tahlequah ok` house_number=`17560` postal_code=`None`
  raw_country='US' -> normalized=`united states`

- raw_name='B+ Retail Inc' -> exact=`b retail inc` suffix_stripped=`b retail`
  raw_address='1712 Montebello Avenue, Phoenix, AZ' -> normalized=`1712 montebello avenue phoenix az` house_number=`1712` postal_code=`None`
  raw_country='US' -> normalized=`united states`

- raw_name='Christ Chapel' -> exact=`christ chapel` suffix_stripped=`christ chapel`
  raw_address='2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD' -> normalized=`2100 cameron drive unit apartment g dundalk md` house_number=`2100` postal_code=`None`
  raw_country='US' -> normalized=`united states`

- raw_name='Prabhav Business Center' -> exact=`prabhav business center` suffix_stripped=`prabhav business center`
  raw_address='797, Lake Town Block A, Kolkata, Howrah, West Bengal' -> normalized=`797 lake town block a kolkata howrah west bengal` house_number=`797` postal_code=`None`
  raw_country='India' -> normalized=`india`

- raw_name='Custom Wealth Services LLC' -> exact=`custom wealth services llc` suffix_stripped=`custom wealth services`
  raw_address='OH, Columbus, 5559 Orville Avenue' -> normalized=`oh columbus 5559 orville avenue` house_number=`None` postal_code=`5559`
  raw_country='US' -> normalized=`united states`

- raw_name='Consulting Nyasa Nursing Private Limited' -> exact=`consulting nyasa nursing private limited` suffix_stripped=`consulting nyasa nursing`
  raw_address='2505, Tower 1, Oakwood, Runwal Greens, Mulund Goreagon Link Road, Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra' -> normalized=`2505 tower 1 oakwood runwal greens mulund goreagon link road near fortis hospital bhandup west mumbai maharashtra` house_number=`2505` postal_code=`None`
  raw_country='India' -> normalized=`india`

- raw_name='Nexus Anchor Rain' -> exact=`nexus anchor rain` suffix_stripped=`nexus anchor rain`
  raw_address='1111 Church Street, Unit 2007, Nashville, TN' -> normalized=`1111 church street unit 2007 nashville tn` house_number=`1111` postal_code=`None`
  raw_country='US' -> normalized=`united states`

- raw_name='Moore Bitwise Inc' -> exact=`moore bitwise inc` suffix_stripped=`moore bitwise`
  raw_address='337 Oakland Avenue, Michigan City, IN' -> normalized=`337 oakland avenue michigan city in` house_number=`337` postal_code=`None`
  raw_country='US' -> normalized=`united states`

- raw_name='Dermatology Green Medicine' -> exact=`dermatology green medicine` suffix_stripped=`dermatology green medicine`
  raw_address='294 Meadowcreek Drive, Unit Unit 2, Village Of Pewaukee, WI' -> normalized=`294 meadowcreek drive unit unit 2 village of pewaukee wi` house_number=`294` postal_code=`None`
  raw_country='US' -> normalized=`united states`
