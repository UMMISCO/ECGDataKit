# HL7 aECG schema

XML schema of the HL7 Annotated ECG message (ANSI/HL7 V3 ECG R1-2004,
message type `PORT_MT020001`), release 2003-12, unmodified.

- `schema/PORT_MT020001.xsd`: the message schema
- `dt/datatypes.xsd`, `voc/voc.xsd`, `meta/voc.xsd`: the HL7 data types
  and vocabulary it includes

Copied from the schema set distributed by the U.S. Food and Drug
Administration with its aECG tools (https://github.com/FDA/aecg-python,
`src/aecg/data/hl7/2003-12 Schema`).

This product includes software developed by Health Level Seven.

Copyright (c) 2002, 2003, Health Level Seven. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement: This product includes
   software developed by Health Level Seven.

THIS SOFTWARE IS PROVIDED BY HEALTH LEVEL SEVEN, INC. AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
