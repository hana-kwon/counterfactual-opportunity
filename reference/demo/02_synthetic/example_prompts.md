# Example prompts built from synthetic profiles

These use the same templates as the real runs (src/prompts_grant.py, src/prompts_lending.py). All values are synthetic.

## Grant competition, 1|XR|LA

```
SYSTEM: You are an analyst who evaluates applicants to a business support program in Nigeria. Use only the information provided. Answer with a single JSON object and nothing else.

Applicant to the YouWiN! business plan competition, Nigeria, application made in late 2011. The competition can award this applicant a grant averaging about US$50,000, paid in installments during the following year.

Applicant profile
- Owner: female, age 44, divorced or separated, 1 children, education: polytechnic or college of education
- State of residence: Enugu
- Application: new business; sector: shop or restaurant
- Training taken: business accounting yes, marketing no, computer skills no, vocational yes
- Currently employed outside the firm: no; has lived or worked abroad: yes
- Household owns: 0 computer(s), 1 car(s); internet at home: yes
- Years of work experience: 10
- Current loans: not stated
- What had prevented the applicant from starting or expanding a business: lack of funds

Questions:
Q1. Expected total employment in the applicant's firm three years from now, counting the owner, if the applicant does not receive the grant (0 if no firm is operating).
Q2. Expected total employment in the applicant's firm three years from now, counting the owner, if the applicant receives the grant.
Q3. Does the profile state the facts needed to judge how much the grant would change employment? Answer "sufficient" or "insufficient".

Return exactly: {"emp_without": number, "emp_with": number, "information": "sufficient" or "insufficient"}
```

## Grant competition, 1|X|F

```
SYSTEM: You are an analyst who evaluates applicants to a business support program in Nigeria. Use only the information provided. Answer with a single JSON object and nothing else.

Applicant to the YouWiN! business plan competition, Nigeria, application made in late 2011. The competition can award this applicant a grant averaging about US$50,000, paid in installments during the following year.

Applicant profile
- Owner: female, age 44, divorced or separated, 1 children, education: polytechnic or college of education
- State of residence: Enugu
- Application: new business; sector: shop or restaurant
- Training taken: business accounting yes, marketing no, computer skills no, vocational yes
- Currently employed outside the firm: no; has lived or worked abroad: yes
- Household owns: 0 computer(s), 1 car(s); internet at home: yes
- Years of work experience: 10

Question: What is the probability (0-100) that this applicant obtains at least US$10,000 of outside financing (a loan or an equity investment) within the next 12 months if it does not receive this grant?

Return exactly: {"p_outside_financing": number}
```

## Lending, 1000|P_RB

```
SYSTEM: You are an analyst who advises a microfinance lender. Use only the information provided. Answer with a single JSON object and nothing else.

Borrower of a nonprofit microfinance lender in Egypt, 2016-2017. The borrower runs a small business and has repaid at least three earlier loans from this lender.

Borrower profile
- Owner age: 40-49
- Owner gender: male
- Owner education: some or completed post-secondary, college or advanced degree
- Business activity: services, retail sales
- Years in business: 11-20
- Business has staff: yes
- Business is registered: no
- Previous loan from this lender (Egyptian pounds): 15,000 or more
- Monthly profit of the main business before the offer (Egyptian pounds): 2,500-4,999
- Monthly revenue of the main business before the offer (Egyptian pounds): 5,000-9,999

Other financing
- Has other loans now: yes (from: a rotating savings group (ROSCA))
- Lack of credit as a constraint on current operations: an average constraint
- Lists lack of credit among the constraints that will limit the business over the long term: yes

Planned use of a larger loan
- Planned uses of a larger loan (owner's answer, several allowed): paying off other debt, other (details not shared)
- Primary planned use: paying off other debt

The lender can offer this borrower one of two loans:
- Normal offer: a loan about twice the size of the borrower's previous loan from this lender.
- Large offer: a loan about four times the size of the previous loan.
Annual interest is 14-17 percent. The borrower chooses a term of 6 to 24 months (12 months is most common). Late installments are charged a penalty. A borrower who declines an offer takes no loan from this lender in this cycle, so that borrower has no late installments.

Goal: The lender wants to know for which borrowers the large offer, compared with the normal offer, would raise the borrower's business profit the most.

Answer each question for the normal offer and for the large offer. Decimals are allowed. Months are counted from the date the lender makes the offer.
Q1. Probability (0-100) that the borrower accepts the offer.
Q2. If the borrower accepts, probability (0-100) that at least one monthly installment is paid late during the loan.
Q3. Expected average monthly profit in Egyptian pounds about 20 months after the offer, summed over all businesses the owner runs at that time (profit = what the businesses earn after all expenses except payments to the owners; 0 if the owner runs no business then), whether or not the borrower accepts the offer.
Q4. The same quantity as Q3, about 30 months after the offer.
Q5. Does the profile state the facts needed to judge how the large offer would change this borrower's profit and repayment? Answer "sufficient" or "insufficient".

Return exactly: {"normal": {"accept": number, "late_if_accepted": number, "profit_20m": number, "profit_30m": number}, "large": {"accept": number, "late_if_accepted": number, "profit_20m": number, "profit_30m": number}, "information": "sufficient" or "insufficient"}
```

## Lending, 1000|S_RBP

```
SYSTEM: You are an analyst who advises a microfinance lender. Use only the information provided. Answer with a single JSON object and nothing else.

Borrower of a nonprofit microfinance lender in Egypt, 2016-2017. The borrower runs a small business and has repaid at least three earlier loans from this lender.

Borrower profile
- Owner age: 40-49
- Owner gender: male
- Owner education: some or completed post-secondary, college or advanced degree
- Business activity: services, retail sales
- Years in business: 11-20
- Business has staff: yes
- Business is registered: no
- Previous loan from this lender (Egyptian pounds): 15,000 or more
- Monthly profit of the main business before the offer (Egyptian pounds): 2,500-4,999
- Monthly revenue of the main business before the offer (Egyptian pounds): 5,000-9,999

Other financing
- Has other loans now: yes (from: a rotating savings group (ROSCA))
- Lack of credit as a constraint on current operations: an average constraint
- Lists lack of credit among the constraints that will limit the business over the long term: yes

Planned use of a larger loan
- Planned uses of a larger loan (owner's answer, several allowed): paying off other debt, other (details not shared)
- Primary planned use: paying off other debt

How much the owner agrees with each statement
- Synthetic statement 1 (placeholder for the original survey item): disagree
- Synthetic statement 2 (placeholder for the original survey item): strongly disagree
- Synthetic statement 3 (placeholder for the original survey item): neither agree nor disagree
- Synthetic statement 4 (placeholder for the original survey item): strongly disagree
- Synthetic statement 5 (placeholder for the original survey item): agree
- Synthetic statement 6 (placeholder for the original survey item): agree
- Synthetic statement 7 (placeholder for the original survey item): disagree
- Synthetic statement 8 (placeholder for the original survey item): agree
- Synthetic statement 9 (placeholder for the original survey item): strongly disagree
- Synthetic statement 10 (placeholder for the original survey item): agree
- Synthetic statement 11 (placeholder for the original survey item): neither agree nor disagree
- Synthetic statement 12 (placeholder for the original survey item): disagree
- Synthetic statement 13 (placeholder for the original survey item): agree
- Synthetic statement 14 (placeholder for the original survey item): neither agree nor disagree
- Synthetic statement 15 (placeholder for the original survey item): strongly agree
- Synthetic statement 16 (placeholder for the original survey item): neither agree nor disagree
- Synthetic statement 17 (placeholder for the original survey item): strongly agree
- Synthetic statement 18 (placeholder for the original survey item): strongly disagree
- Synthetic statement 19 (placeholder for the original survey item): neither agree nor disagree
- Synthetic statement 20 (placeholder for the original survey item): strongly disagree
- Synthetic statement 21 (placeholder for the original survey item): agree
- Synthetic statement 22 (placeholder for the original survey item): strongly disagree
- Synthetic statement 23 (placeholder for the original survey item): strongly disagree
- Synthetic statement 24 (placeholder for the original survey item): agree
- Synthetic statement 25 (placeholder for the original survey item): strongly agree
- Synthetic statement 26 (placeholder for the original survey item): strongly agree
- Synthetic statement 27 (placeholder for the original survey item): strongly disagree
- Synthetic statement 28 (placeholder for the original survey item): strongly disagree
- Synthetic statement 29 (placeholder for the original survey item): disagree
- Synthetic statement 30 (placeholder for the original survey item): disagree
- Synthetic statement 31 (placeholder for the original survey item): agree
- Synthetic statement 32 (placeholder for the original survey item): disagree
- Synthetic statement 33 (placeholder for the original survey item): neither agree nor disagree
- Synthetic statement 34 (placeholder for the original survey item): agree
- Synthetic statement 35 (placeholder for the original survey item): agree
- Synthetic statement 36 (placeholder for the original survey item): strongly disagree
- Synthetic statement 37 (placeholder for the original survey item): strongly disagree
- Synthetic statement 38 (placeholder for the original survey item): strongly disagree
- Synthetic statement 39 (placeholder for the original survey item): neither agree nor disagree
- Synthetic statement 40 (placeholder for the original survey item): strongly disagree
- Synthetic statement 41 (placeholder for the original survey item): disagree
- Synthetic statement 42 (placeholder for the original survey item): strongly agree
- Synthetic statement 43 (placeholder for the original survey item): agree
- Synthetic statement 44 (placeholder for the original survey item): agree
- Synthetic statement 45 (placeholder for the original survey item): agree
- Synthetic statement 46 (placeholder for the original survey item): strongly agree
- Synthetic statement 47 (placeholder for the original survey item): agree
- Synthetic statement 48 (placeholder for the original survey item): neither agree nor disagree
- Synthetic statement 49 (placeholder for the original survey item): neither agree nor disagree
- Synthetic statement 50 (placeholder for the original survey item): neither agree nor disagree

The lender can offer this borrower one of two loans:
- Normal offer: a loan about twice the size of the borrower's previous loan from this lender.
- Large offer: a loan about four times the size of the previous loan.
Annual interest is 14-17 percent. The borrower chooses a term of 6 to 24 months (12 months is most common). Late installments are charged a penalty. A borrower who declines an offer takes no loan from this lender in this cycle, so that borrower has no late installments.

Goal: The lender wants to know for which borrowers the large offer, compared with the normal offer, would lower the chance of late repayment the most, or raise it the least.

Answer each question for the normal offer and for the large offer. Decimals are allowed. Months are counted from the date the lender makes the offer.
Q1. Probability (0-100) that the borrower accepts the offer.
Q2. If the borrower accepts, probability (0-100) that at least one monthly installment is paid late during the loan.
Q3. Expected average monthly profit in Egyptian pounds about 20 months after the offer, summed over all businesses the owner runs at that time (profit = what the businesses earn after all expenses except payments to the owners; 0 if the owner runs no business then), whether or not the borrower accepts the offer.
Q4. The same quantity as Q3, about 30 months after the offer.
Q5. Does the profile state the facts needed to judge how the large offer would change this borrower's profit and repayment? Answer "sufficient" or "insufficient".

Return exactly: {"normal": {"accept": number, "late_if_accepted": number, "profit_20m": number, "profit_30m": number}, "large": {"accept": number, "late_if_accepted": number, "profit_20m": number, "profit_30m": number}, "information": "sufficient" or "insufficient"}
```

