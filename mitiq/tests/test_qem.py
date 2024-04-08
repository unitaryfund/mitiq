# Copyright (C) Unitary Fund
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

# Check that mitiq.qem maintains backward compatibility
# for the techniques inside it (i.e. you can still call
# mitiq.zne )
from mitiq import zne 
from mitiq.zne import scaling
from mitiq.qem import zne