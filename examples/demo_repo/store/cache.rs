use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct Entry<V> {
    value: V,
    born: Instant,
}

pub struct Expiring<K, V> {
    items: HashMap<K, Entry<V>>,
    ttl: Duration,
}

impl<K: std::hash::Hash + Eq, V: Clone> Expiring<K, V> {
    pub fn new(ttl: Duration) -> Self {
        Self { items: HashMap::new(), ttl }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        let entry = self.items.get(key)?;
        if entry.born.elapsed() > self.ttl {
            self.items.remove(key);
            return None;
        }
        Some(entry.value.clone())
    }

    pub fn put(&mut self, key: K, value: V) {
        self.items.insert(key, Entry { value, born: Instant::now() });
    }
}
