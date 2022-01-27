use std::ops::{Index, IndexMut};

pub type Vector2i = Vector2<i32>;
pub type Vector2f = Vector2<f32>;
pub type Vector3i = Vector3<i32>;
pub type Vector3f = Vector3<f32>;

pub trait Vector<T> : Index<usize> + IndexMut<usize> {
}

pub struct Vector2<T> {
    x: T,
    y: T,
}

impl<T> Vector2<T> {
    pub fn new(x: T, y: T) -> Self {
        Vector2 { x, y }
    }
}

impl<T> Vector<T> for Vector2<T> {
}

impl<T> Index<usize> for Vector2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index provided non-existent"),
        }
    }
}

impl<T> IndexMut<usize> for Vector2<T> {

    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index provided non-existent"),
        }
    }
}


pub struct Vector3<T> {
    x: T,
    y: T,
    z: T,
}

impl<T> Vector3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Vector3 { x, y, z }
    }
}

impl<T> Vector<T> for Vector3<T> {
}

impl<T> Index<usize> for Vector3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index provided non-existent"),
        }
    }
}

impl<T> IndexMut<usize> for Vector3<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index provided non-existent")
        }
    }
}

#[cfg(test)]
mod test_vector_ops {
    use super::{Vector2, Vector3, Vector2f, Vector2i};

    #[test]
    fn indexing() {
        let vec2 = Vector2 { x: 2, y: 3 };
        assert_eq!(vec2[0], 2);
        assert_eq!(vec2[1], 3);

        let vec3 = Vector3 { x: 4, y: 5, z: 6 };
        assert_eq!(vec3[0], 4);
        assert_eq!(vec3[1], 5);
        assert_eq!(vec3[2], 6);
    }

    #[test]
    fn creation() {
        let vec2 = Vector2::new(1, 2);
        assert_eq!(vec2.x, 1);
        assert_eq!(vec2.y, 2);

        let vec3 = Vector3::new(1, 2, 3);
        assert_eq!(vec3.x, 1);
        assert_eq!(vec3.y, 2);
        assert_eq!(vec3.z, 3);
    }
}