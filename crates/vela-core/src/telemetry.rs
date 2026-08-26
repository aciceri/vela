//! Named force components, published every step.
//!
//! This is not debug scaffolding. In a simulator whose purpose is to be
//! *physical*, the breakdown is the product: a total driving force tells you
//! nothing about why the boat is slow, and the same breakdown is what a
//! component-wise regression test compares and what a trim display shows.
//!
//! # Keys are API
//!
//! Keys are `module.item.quantity`, lower case, dot separated. They are a
//! public contract: renaming one breaks every consumer silently, because a
//! missing key reads as an absent force rather than an error. Add keys freely;
//! rename them never.

/// A flat set of named scalars, rebuilt each step.
///
/// Backed by a vector rather than a map because the key set is small, fixed at
/// compile time, and written far more often than it is looked up — and because
/// insertion order is the natural reporting order, which a hash map would
/// scramble and a sorted map would reorder into something no reader chose.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Telemetry {
    entries: Vec<(&'static str, f64)>,
}

impl Telemetry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a value, replacing any previous value for the same key.
    ///
    /// Replacing rather than appending keeps a module that publishes
    /// conditionally from producing duplicate keys whose reader sees whichever
    /// came first.
    pub fn set(&mut self, key: &'static str, value: f64) {
        match self.entries.iter_mut().find(|(name, _)| *name == key) {
            Some(entry) => entry.1 = value,
            None => self.entries.push((key, value)),
        }
    }

    #[must_use]
    pub fn get(&self, key: &str) -> Option<f64> {
        self.entries
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| *value)
    }

    /// Entries in the order they were first recorded.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, f64)> + '_ {
        self.entries.iter().copied()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Empties the set while keeping the allocation, so that stepping a
    /// simulation does not allocate once it has warmed up.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setting_a_key_twice_replaces_rather_than_duplicates() {
        let mut telemetry = Telemetry::new();
        telemetry.set("aero.driving", 1.0);
        telemetry.set("aero.driving", 2.0);
        assert_eq!(telemetry.len(), 1);
        assert_eq!(telemetry.get("aero.driving"), Some(2.0));
    }

    #[test]
    fn insertion_order_is_preserved() {
        let mut telemetry = Telemetry::new();
        telemetry.set("b", 1.0);
        telemetry.set("a", 2.0);
        let keys: Vec<_> = telemetry.iter().map(|(key, _)| key).collect();
        assert_eq!(keys, vec!["b", "a"]);
    }

    #[test]
    fn an_absent_key_is_none_rather_than_zero() {
        // The distinction matters: zero is a force, absent is a module that did
        // not run.
        let telemetry = Telemetry::new();
        assert_eq!(telemetry.get("nothing.here"), None);
    }

    #[test]
    fn clearing_keeps_the_set_usable() {
        let mut telemetry = Telemetry::new();
        telemetry.set("a", 1.0);
        telemetry.clear();
        assert!(telemetry.is_empty());
        telemetry.set("a", 3.0);
        assert_eq!(telemetry.get("a"), Some(3.0));
    }
}
