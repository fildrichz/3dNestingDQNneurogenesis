# EMS-Based Reward Shaping

## The Key Insight

**EMS quality predicts future packing success better than current utilization!**

### Example:

**State A:**
- 30 items placed, util = 0.60
- Created 50 tiny EMS (fragmented space)
- Largest EMS: 100×100×50
- Hard to fit remaining items

**State B:**
- 30 items placed, util = 0.60
- Created 10 large EMS (compact packing)
- Largest EMS: 400×400×400
- Easy to fit remaining items

**Current reward:** Both get same reward (0.60 util)

**EMS-aware reward:** State B gets higher reward (better future potential)

---

## Why EMS Quality Matters

### Bad Packing Creates Fragmentation:

```
┌─────────────────┐
│  ┌──┐  ┌─┐ ┌──┐│  Many small EMS
│  └──┘  └─┘ └──┘│  Hard to use
│ ┌─┐ ┌────┐ ┌─┐ │  Wasted space
│ └─┘ └────┘ └─┘ │
└─────────────────┘
Result: Can't fit more items despite low utilization
```

### Good Packing Preserves Large Spaces:

```
┌─────────────────┐
│┌──────┐         │  Few large EMS
││      │  Large  │  Easy to use
│└──────┘  EMS    │  Flexible
│┌────┐           │
│└────┘           │
└─────────────────┘
Result: Can fit many more items
```

---

## EMS Quality Metrics

### 1. **Largest EMS Volume** (Simple, Effective)

```python
largest_ems_volume = max(ems.volume() for ems in bin.ems_list)
normalized = largest_ems_volume / bin_volume
```

**Intuition:** Maintain at least one big space for future items

**Pros:**
- ✅ Simple to compute
- ✅ Directly measures flexibility
- ✅ Correlates with future success

**Cons:**
- Ignores other EMS
- Doesn't capture fragmentation fully

---

### 2. **Average Usable EMS Volume** (Better)

```python
# Count EMS that can fit typical items
item_size_threshold = median_item_volume
usable_ems = [e for e in bin.ems_list if e.volume() > item_size_threshold]

if usable_ems:
    avg_usable_volume = sum(e.volume() for e in usable_ems) / len(usable_ems)
else:
    avg_usable_volume = 0

normalized = avg_usable_volume / bin_volume
```

**Intuition:** Want many medium-large spaces, not one huge + many tiny

**Pros:**
- ✅ Captures fragmentation
- ✅ Considers all useful space
- ✅ More robust than max alone

---

### 3. **Weighted EMS Quality** (Most Sophisticated)

```python
ems_quality = 0
for ems in bin.ems_list:
    vol = ems.volume()

    # Larger spaces weighted more heavily
    if vol > large_threshold:
        ems_quality += vol * 2.0  # Large EMS very valuable
    elif vol > medium_threshold:
        ems_quality += vol * 1.0  # Medium EMS valuable
    else:
        ems_quality += vol * 0.1  # Tiny EMS not very useful

normalized = ems_quality / bin_volume
```

**Intuition:** Different sized EMS have different utility

---

### 4. **Number of Usable EMS** (Complementary)

```python
usable_count = sum(1 for e in bin.ems_list if e.volume() > threshold)
normalized = usable_count / max_possible_ems
```

**Intuition:** Want enough options for placement

---

## Potential-Based Shaping (Theoretically Sound)

To ensure we don't change the optimal policy, use **potential-based shaping**:

### Theory (Ng et al., 1999)

```python
# Define potential function
Φ(s) = utilization(s) + β * ems_quality(s)

# Shaped reward
F(s, a, s') = r(s,a,s') + γ*Φ(s') - Φ(s)
```

**Theorem:** This doesn't change optimal policy if Φ is well-defined!

### Implementation

```python
# Current state potential
current_util = self.total_placed_volume / total_volume
current_ems_quality = self._compute_ems_quality(target_bin)
potential_current = current_util + 0.2 * current_ems_quality

# After placement potential
next_util = (self.total_placed_volume + item_volume) / total_volume
next_ems_quality = self._compute_ems_quality(target_bin)  # After update_ems
potential_next = next_util + 0.2 * next_ems_quality

# Potential-based shaped reward
reward = (self.gamma * potential_next) - potential_current
```

**β = 0.2** is the weight (tune this!)

---

## Recommended EMS Quality Function

Start simple, then refine:

### Version 1: Largest EMS (Simplest)

```python
def compute_ems_quality(bin):
    if not bin.ems_list:
        return 0.0

    largest_volume = max(ems.volume() for ems in bin.ems_list)
    return largest_volume / bin.volume()
```

### Version 2: Top-K EMS (Better)

```python
def compute_ems_quality(bin, k=5):
    if not bin.ems_list:
        return 0.0

    # Take top-k largest EMS
    volumes = sorted([ems.volume() for ems in bin.ems_list], reverse=True)
    top_k_volumes = volumes[:min(k, len(volumes))]

    avg_top_k = sum(top_k_volumes) / len(top_k_volumes)
    return avg_top_k / bin.volume()
```

**Intuition:** Maintain several large spaces, not just one

### Version 3: Weighted by Size (Most Robust)

```python
def compute_ems_quality(bin, item_volumes):
    if not bin.ems_list:
        return 0.0

    # Threshold: median remaining item size
    if len(item_volumes) > 0:
        threshold = np.median(item_volumes)
    else:
        threshold = 1000  # Default

    quality = 0
    for ems in bin.ems_list:
        vol = ems.volume()
        if vol > threshold * 3:
            quality += vol * 2.0  # Very valuable
        elif vol > threshold:
            quality += vol * 1.0  # Valuable
        else:
            quality += vol * 0.1  # Less useful

    return quality / bin.volume()
```

---

## Advantages of EMS-Based Shaping

### 1. **Domain Knowledge** ✅
- Uses structure specific to EMS-based packing
- Not generic heuristic (like "place low")
- Directly related to problem

### 2. **Theoretically Sound** ✅
- Potential-based shaping preserves optimality
- Doesn't bias toward wrong solution
- Just speeds up learning

### 3. **Addresses Credit Assignment** ✅
- Good placements → better EMS quality → immediate feedback
- Bad placements → fragmentation → immediate penalty
- Don't need to wait for episode end

### 4. **Scales to Any Episode Length** ✅
- Works for 50-item episodes
- Works for 200-item episodes
- Immediate signal regardless

### 5. **Interpretable** ✅
- Can visualize EMS quality over time
- Debug why certain actions preferred
- Understand agent's reasoning

---

## Comparison to Other Shaping

### Current (Pure Utilization):
```python
reward = (gamma * util_next) - util_prev
```
- ✅ Simple
- ❌ Doesn't distinguish good/bad placements at same util
- ❌ No fragmentation awareness

### Height/Slack Penalties (What I suggested earlier):
```python
reward = util_increase - height_penalty - slack_penalty
```
- ✅ Immediate feedback
- ❌ Might bias toward local optimum
- ❌ Hard to tune weights
- ❌ May not correlate with true objective

### EMS-Based (Your idea):
```python
potential = util + β * ems_quality
reward = (gamma * potential_next) - potential_current
```
- ✅ Immediate feedback
- ✅ Theoretically sound (potential-based)
- ✅ Directly predicts future success
- ✅ Domain-specific knowledge
- ✅ One parameter to tune (β)

---

## Implementation

```python
def _compute_ems_quality(self, bin):
    """Compute EMS quality for potential-based shaping."""
    if not bin.ems_list:
        return 0.0

    # Version 2: Top-5 average
    volumes = sorted([ems.volume() for ems in bin.ems_list], reverse=True)
    top_5 = volumes[:min(5, len(volumes))]
    avg_volume = sum(top_5) / len(top_5)

    return avg_volume / bin.volume()

# In step() function (line 345):
def step(self, action):
    # ... existing code ...

    # BEFORE placement
    util_prev = self.total_placed_volume / total_available_volume
    ems_quality_prev = self._compute_ems_quality(target_bin)
    potential_prev = util_prev + 0.2 * ems_quality_prev

    # Place item (this updates EMS)
    ok = target_bin.place_at_ems(ems, size, weight=weight, item_id=item_id)

    # AFTER placement
    util_next = self.total_placed_volume / total_available_volume
    ems_quality_next = self._compute_ems_quality(target_bin)
    potential_next = util_next + 0.2 * ems_quality_next

    # Potential-based shaped reward
    reward = (self.gamma * potential_next) - potential_prev
```

---

## Tuning β (EMS Weight)

Start with β = 0.1 to 0.3:

```python
β = 0.1  # Conservative: 10% EMS quality, 90% utilization
β = 0.2  # Balanced: 20% EMS quality, 80% utilization
β = 0.3  # Aggressive: 30% EMS quality, 70% utilization
```

Monitor:
- Does final utilization improve?
- Do visualizations show less fragmentation?
- Does completion rate increase?

If β too high: Agent might sacrifice utilization for EMS quality
If β too low: Doesn't help much

---

## Expected Impact

### With EMS-Based Shaping:

**Episode step 15:**
- Placement A: +0.001 util, creates 10 tiny EMS → Low EMS quality
- Placement B: +0.001 util, maintains 3 large EMS → High EMS quality
- **Now agent can distinguish them immediately!**

**Result:**
- ✅ Better early decisions
- ✅ Less fragmentation
- ✅ Higher final utilization
- ✅ Better credit assignment
- ✅ Works for any episode length

---

## Recommendation

**Implement EMS-based potential shaping:**

1. Start with simple version (top-5 average EMS volume)
2. Set β = 0.2
3. Train and compare to baseline
4. Tune β if needed
5. Try weighted version if simple doesn't work

This addresses your credit assignment concern while being theoretically sound and using domain knowledge specific to your EMS implementation!

**This is a really good idea!** Much better than generic heuristics like height penalties.
