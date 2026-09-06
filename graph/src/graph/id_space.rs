//! Validating a graph's node id space across a batch of ingested ids.
//!
//! ## The invariant, and why the graph cannot check it alone
//!
//! `Graph` tracks node liveness with `node_count` and `deleted_nodes` and no
//! live set, so the only boundary it can offer is
//! `node_count + deleted_nodes.len()`. That is the boundary *if the id space is
//! dense*, and `max_node_id` is the same arithmetic — so an assertion written in
//! terms of them checks one derivation of the state against another derivation
//! of the same state, and passes on a graph that is already wrong.
//!
//! Density holds between batches, not within one. The effects apply path ingests
//! records grouped by shape rather than ordered by id, so a create of 500..600
//! may precede one of 0..500; between them the graph holds 100 live nodes whose
//! highest id is 599 and reports its boundary as 100.
//!
//! ## What this holds
//!
//! The boundary as it stood before the batch, and every id at or above it that
//! the batch has ingested. That is enough to state the invariant directly:
//!
//! ```text
//! ingested == [ entry_bound, entry_bound + ingested.len() )
//! ```
//!
//! Two independent things can break it, so [`NodeIdSpace::verify`] checks both.
//!
//! **The shape of what arrived.** Ids ingested at or above the boundary must
//! fill the range from it upward with no hole. An allocator hands out the lowest
//! free id, so it cannot reach an id without having handed out everything below
//! — a batch that leaves a hole did not come from one. This is the check that
//! catches a replica which has missed a buffer: told to create 500..600 when its
//! own boundary is 0, it would otherwise accept an id space its master does not
//! have and collide on its next allocation.
//!
//! **That the graph counted it.** `node_count` is an independent counter, and
//! the same id ingested twice moves it twice while the set absorbs the
//! duplicate. Comparing the graph's own boundary against
//! `entry_bound + ingested.len()` is the only place anything checks that counter
//! against a value not derived from it.
//!
//! Neither subsumes the other: the first misses a duplicate inside an otherwise
//! complete range, the second misses a hole, because the ids in a truncated
//! batch really were counted.
//!
//! Nothing here knows about replication, buffers or a peer engine. It is a
//! statement about one graph and the ids handed to it.

use crate::graph::graph::Graph;
use roaring::RoaringTreemap;

/// Why a batch of ingested node ids does not describe a possible id space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdSpaceError {
    /// A delete named an id at or above the entry boundary that this batch never
    /// created, so it was never allocated here at all.
    NeverIngested(u64),
    /// Ids are missing between the entry boundary and the highest one ingested.
    /// `ingested` many ids should have reached `entry_bound + ingested - 1`.
    Hole {
        entry_bound: u64,
        highest: u64,
        ingested: u64,
    },
    /// The graph's own boundary is not where the ingested ids put it — it
    /// counted something twice, or not at all.
    Miscounted { graph_bound: u64, expected: u64 },
    /// A batch named `u64::MAX`, which has no boundary above it.
    IdOutOfRange(u64),
}

impl std::fmt::Display for IdSpaceError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::NeverIngested(id) => write!(f, "node {id} was never allocated here"),
            Self::Hole {
                entry_bound,
                highest,
                ingested,
            } => write!(
                f,
                "node ids {entry_bound}..={highest} were allocated but only {ingested} of them created"
            ),
            Self::Miscounted {
                graph_bound,
                expected,
            } => write!(
                f,
                "the graph's node id boundary is {graph_bound}, but the ids ingested put it at {expected}"
            ),
            Self::IdOutOfRange(id) => write!(f, "node id {id} is past the end of the id space"),
        }
    }
}

/// A graph's node id space, observed across one batch of ingested ids.
///
/// Built before the batch, told what it creates, and checked once at the end.
/// It decides nothing while the batch runs — [`Graph::create_nodes`] already
/// refuses an id that is live below the boundary — so this records and then
/// judges, rather than gating each step.
pub struct NodeIdSpace {
    /// The boundary as it stood before the batch: ids below it were handed out,
    /// ids at or above it never were.
    entry_bound: u64,
    /// Every id at or above `entry_bound` this batch has ingested. Only grows —
    /// a later delete does not un-allocate an id, it frees one that was.
    ingested: RoaringTreemap,
}

impl NodeIdSpace {
    /// The id space as it stands before a batch.
    ///
    /// `node_count + deleted_nodes_count` is the boundary *because* this is
    /// taken between batches, where the id space is dense. That assumption is
    /// stated once, here, rather than implied by an accessor every caller has to
    /// know not to trust.
    ///
    /// Not `max_node_id() + 1`, though the arithmetic agrees whenever the graph
    /// holds a live node: `max_node_id` returns a 0 *sentinel* when it holds
    /// none, which is indistinguishable from a graph whose highest id is 0 and
    /// reads as "id 0 has been handed out".
    #[must_use]
    pub fn at_entry(g: &Graph) -> Self {
        Self {
            entry_bound: g.node_count() + g.deleted_nodes_count(),
            ingested: RoaringTreemap::new(),
        }
    }

    /// The boundary as it stood before the batch.
    #[must_use]
    pub const fn entry_bound(&self) -> u64 {
        self.entry_bound
    }

    /// Record ids the batch creates.
    ///
    /// Ids below the entry boundary are recycled rather than new, so they are
    /// not the id space growing and are not recorded. [`Graph::create_nodes`]
    /// has already refused any of them that were live.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::IdOutOfRange`] for `u64::MAX`. Nothing can be allocated
    /// above it, and letting it through would wrap the arithmetic that follows.
    pub fn created(
        &mut self,
        nodes: &RoaringTreemap,
    ) -> Result<(), IdSpaceError> {
        if nodes.contains(u64::MAX) {
            return Err(IdSpaceError::IdOutOfRange(u64::MAX));
        }
        // `>= entry_bound` only: a recycled id was already counted in the
        // boundary this started from.
        //
        // The whole-set union is the ordinary case — a batch's creates are
        // either all new or all recycled — and it is a merge rather than a walk.
        // The filtered form is only reached by a batch that mixes the two.
        match nodes.min() {
            None => {}
            Some(min) if min >= self.entry_bound => self.ingested |= nodes,
            Some(_) => self
                .ingested
                .extend(nodes.iter().filter(|&id| id >= self.entry_bound)),
        }
        Ok(())
    }

    /// Check that ids the batch is about to delete were allocated here.
    ///
    /// The other half of "is this node live" — whether it is already in the
    /// recycle bin — is [`Graph::delete_nodes`]'s to refuse, and it needs no
    /// boundary to answer.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::NeverIngested`] for the highest id at or above the entry
    /// boundary that this batch did not create. Above the boundary it was never
    /// allocated before the batch either, so nothing has ever held it.
    pub fn deletable(
        &self,
        nodes: &RoaringTreemap,
    ) -> Result<(), IdSpaceError> {
        match (nodes - &self.ingested)
            .max()
            .filter(|&id| id >= self.entry_bound)
        {
            Some(id) => Err(IdSpaceError::NeverIngested(id)),
            None => Ok(()),
        }
    }

    /// Check that the batch left a possible id space behind.
    ///
    /// # Errors
    ///
    /// [`IdSpaceError::Hole`] if the ingested ids do not fill the range from the
    /// entry boundary upward, and [`IdSpaceError::Miscounted`] if the graph's own
    /// boundary disagrees with where those ids put it. See the module docs for
    /// why both are needed.
    pub fn verify(
        &self,
        g: &Graph,
    ) -> Result<(), IdSpaceError> {
        let ingested = self.ingested.len();
        // `len - 1 == highest - entry_bound` rather than `highest - entry + 1 ==
        // len`: every id in the set is at or above the boundary and the set is
        // non-empty here, so both sides are subtractions that cannot wrap, where
        // the `+ 1` form can.
        if let Some(highest) = self
            .ingested
            .max()
            .filter(|&highest| ingested - 1 != highest - self.entry_bound)
        {
            return Err(IdSpaceError::Hole {
                entry_bound: self.entry_bound,
                highest,
                ingested,
            });
        }

        let graph_bound = g.node_count() + g.deleted_nodes_count();
        let expected = self
            .entry_bound
            .checked_add(ingested)
            .ok_or(IdSpaceError::IdOutOfRange(u64::MAX))?;
        if graph_bound != expected {
            return Err(IdSpaceError::Miscounted {
                graph_bound,
                expected,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{IdSpaceError, NodeIdSpace};
    use crate::graph::graph::Graph;
    use crate::graph::graphblas::test_init::ensure_init;
    use roaring::RoaringTreemap;
    use rustc_hash::FxHashMap;

    fn graph() -> Graph {
        ensure_init();
        Graph::new(64, 64, 0, 0, "t")
    }

    fn ids(v: &[u64]) -> RoaringTreemap {
        v.iter().copied().collect()
    }

    fn range(r: std::ops::Range<u64>) -> RoaringTreemap {
        r.collect()
    }

    /// Create through the graph as the ingest path does, so the graph's own
    /// counters move and `verify` is checked against something real.
    fn create(
        space: &mut NodeIdSpace,
        g: &mut Graph,
        nodes: &RoaringTreemap,
    ) {
        space.created(nodes).expect("in range");
        g.add_reserved_node_count(nodes.len());
        g.create_nodes(nodes, space.entry_bound())
            .expect("the graph accepts it");
    }

    fn delete(
        space: &NodeIdSpace,
        g: &mut Graph,
        nodes: &RoaringTreemap,
    ) -> Result<(), IdSpaceError> {
        space.deletable(nodes)?;
        g.delete_nodes(nodes, &mut FxHashMap::default())
            .map(|_| ())
            .expect("the graph accepts it");
        Ok(())
    }

    #[test]
    fn ids_arriving_out_of_order_still_fill_the_range() {
        // The case a boundary derived from the counter cannot express: the high
        // half arrives first, so mid-batch the graph reports its boundary as 100
        // while id 599 is allocated.
        let mut g = graph();
        let mut space = NodeIdSpace::at_entry(&g);

        create(&mut space, &mut g, &range(500..600));
        assert!(
            space.verify(&g).is_err(),
            "mid-batch the range has a hole, which the graph alone cannot see"
        );

        create(&mut space, &mut g, &range(0..500));
        space.verify(&g).expect("the batch closed the hole");
        assert_eq!(g.node_count(), 600);
    }

    #[test]
    fn a_hole_left_at_the_end_is_reported() {
        // 500..600 and nothing else. An allocator hands out the lowest free id,
        // so it cannot reach 500 without having handed out 0..499 — whoever
        // produced this was not working from the same id space.
        let mut g = graph();
        let mut space = NodeIdSpace::at_entry(&g);
        create(&mut space, &mut g, &range(500..600));

        let err = space.verify(&g).expect_err("must report the hole");
        assert_eq!(
            err,
            IdSpaceError::Hole {
                entry_bound: 0,
                highest: 599,
                ingested: 100,
            }
        );
    }

    #[test]
    fn an_id_ingested_twice_is_caught_by_the_count() {
        // The set absorbs the duplicate, so the range still looks whole — but
        // `node_count` moved twice. Only the counter check sees this.
        let mut g = graph();
        let mut space = NodeIdSpace::at_entry(&g);
        create(&mut space, &mut g, &range(0..6));

        // The same id again. The graph cannot refuse it: it is at or above the
        // entry boundary, so `create_nodes` reads it as fresh.
        create(&mut space, &mut g, &ids(&[5]));

        let err = space.verify(&g).expect_err("must report the miscount");
        assert_eq!(
            err,
            IdSpaceError::Miscounted {
                graph_bound: 7,
                expected: 6,
            }
        );
    }

    #[test]
    fn a_recycled_id_does_not_grow_the_space() {
        // Ids below the entry boundary were counted in the boundary this started
        // from, so recreating one leaves the range and the count unchanged.
        let mut g = graph();
        let mut space = NodeIdSpace::at_entry(&g);
        create(&mut space, &mut g, &ids(&[0, 1]));
        delete(&space, &mut g, &ids(&[0])).expect("deleting what this batch created");
        space.verify(&g).expect("whole");

        let mut space = NodeIdSpace::at_entry(&g);
        assert_eq!(space.entry_bound(), 2, "the bin is part of the boundary");
        create(&mut space, &mut g, &ids(&[0]));
        assert_eq!(space.entry_bound(), 2, "and id 0 did not extend it");
        space.verify(&g).expect("whole again");
    }

    #[test]
    fn creating_deleting_and_recreating_one_id_in_a_batch() {
        // Three commits of one query reach a replica as a single buffer, and the
        // allocator hands the freed id straight back, so `C(0) · D(0) · C(0)` is
        // legitimate. `ingested` does not shrink on the delete, so the recreate
        // adds nothing and the range stays whole.
        let mut g = graph();
        let mut space = NodeIdSpace::at_entry(&g);
        create(&mut space, &mut g, &ids(&[0]));
        delete(&space, &mut g, &ids(&[0])).expect("delete");
        create(&mut space, &mut g, &ids(&[0]));
        assert_eq!(g.node_count(), 1);
        space.verify(&g).expect("whole");
    }

    #[test]
    fn a_cancelled_reservation_arrives_as_a_pair() {
        // What a create-then-delete in one segment ships: both ids created, then
        // one deleted. The id space grew by two and one of them is free.
        let mut g = graph();
        let mut space = NodeIdSpace::at_entry(&g);
        create(&mut space, &mut g, &ids(&[0, 1]));
        delete(&space, &mut g, &ids(&[0])).expect("delete");
        space.verify(&g).expect("whole");
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.deleted_nodes_count(), 1);
    }

    #[test]
    fn deleting_an_id_never_ingested_is_refused() {
        let mut g = graph();
        let mut space = NodeIdSpace::at_entry(&g);
        create(&mut space, &mut g, &ids(&[0, 1]));

        let err = delete(&space, &mut g, &ids(&[7])).expect_err("7 was never allocated");
        assert_eq!(err, IdSpaceError::NeverIngested(7));
    }

    #[test]
    fn the_first_id_zero_is_not_read_as_already_handed_out() {
        // `max_node_id()` returns 0 for an empty graph, so a boundary taken from
        // it reads as "id 0 has been handed out" and refuses this.
        let mut g = graph();
        let mut space = NodeIdSpace::at_entry(&g);
        assert_eq!(space.entry_bound(), 0, "nothing has been handed out");
        create(&mut space, &mut g, &ids(&[0]));
        space.verify(&g).expect("whole");
    }

    #[test]
    fn the_last_id_is_refused_rather_than_wrapping_the_arithmetic() {
        let g = graph();
        let mut space = NodeIdSpace::at_entry(&g);
        let err = space
            .created(&ids(&[u64::MAX]))
            .expect_err("the top of the id space is not creatable");
        assert_eq!(err, IdSpaceError::IdOutOfRange(u64::MAX));
    }
}
