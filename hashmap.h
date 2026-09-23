/****************************************************************************
 *
 * MODULE:       r.hydro.anuga
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Open-addressing hash map from 64-bit keys to 64-bit values,
 *               used for mesh nodes, edges and leaf lookup.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 *****************************************************************************/

#ifndef R_HYDRO_ANUGA_HASHMAP_H
#define R_HYDRO_ANUGA_HASHMAP_H

#include <stdint.h>

#include <grass/gis.h>

/* Key reserved as "empty slot"; callers never use it. */
#define HASHMAP_EMPTY UINT64_MAX

struct hashmap {
    uint64_t *keys;
    int64_t *values;
    uint64_t mask; /* capacity - 1, capacity a power of two */
    long size;
};

static inline uint64_t hashmap_mix(uint64_t k)
{
    /* splitmix64 finaliser. */
    k ^= k >> 30;
    k *= 0xbf58476d1ce4e5b9ULL;
    k ^= k >> 27;
    k *= 0x94d049bb133111ebULL;
    k ^= k >> 31;

    return k;
}

/* Capacity for at least `expected` entries at a load factor below 0.5. */
static inline void hashmap_init(struct hashmap *h, long expected)
{
    uint64_t cap = 16, i;

    while (cap < (uint64_t)expected * 2)
        cap <<= 1;
    h->keys = G_malloc(cap * sizeof(uint64_t));
    h->values = G_malloc(cap * sizeof(int64_t));
    for (i = 0; i < cap; i++)
        h->keys[i] = HASHMAP_EMPTY;
    h->mask = cap - 1;
    h->size = 0;
}

static inline void hashmap_free(struct hashmap *h)
{
    G_free(h->keys);
    G_free(h->values);
    h->keys = NULL;
    h->values = NULL;
}

/* Slot index holding key, or the empty slot where it would go. */
static inline uint64_t hashmap_slot(const struct hashmap *h, uint64_t key)
{
    uint64_t i = hashmap_mix(key) & h->mask;

    while (h->keys[i] != HASHMAP_EMPTY && h->keys[i] != key)
        i = (i + 1) & h->mask;

    return i;
}

/* Value for key, or -1 if absent (values stored are never negative). */
static inline int64_t hashmap_get(const struct hashmap *h, uint64_t key)
{
    uint64_t i = hashmap_slot(h, key);

    return h->keys[i] == key ? h->values[i] : -1;
}

/* Insert key -> value if absent. Returns the value now stored. The map is
 * sized once by hashmap_init; exceeding half its capacity is a bug. */
static inline int64_t hashmap_put_new(struct hashmap *h, uint64_t key,
                                      int64_t value)
{
    uint64_t i = hashmap_slot(h, key);

    if (h->keys[i] == key)
        return h->values[i];
    if ((uint64_t)(h->size + 1) * 2 > h->mask + 1)
        G_fatal_error("Internal error: hash map capacity exceeded");
    h->keys[i] = key;
    h->values[i] = value;
    h->size++;

    return value;
}

#endif /* R_HYDRO_ANUGA_HASHMAP_H */
