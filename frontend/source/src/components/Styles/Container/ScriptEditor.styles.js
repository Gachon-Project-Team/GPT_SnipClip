import styled from 'styled-components';

export const Container = styled.div`
  padding: 5px;
  background-color: transparent;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  flex: 6;
  width: 100%;
  min-width: 0;
  /* border: solid purple 3px; */
`;

export const TextField = styled.textarea`
  width: 100%;
  font-size: 16px;
  color: black;
  background-color: ${({ editable }) => (editable ? 'white' : '#eee')};
  caret-color: black;
  margin-bottom: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
  padding: 8px;
  resize: none;
  line-height: 1.5;
  height: auto;
  min-height: 3.5rem;
`;

export const SmallButton = styled.button`
  align-self: flex-end;
  padding: 6px 12px;
  font-size: 14px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;

  &:hover {
    background-color: #0056b3;
  }
`;
export const CharCounter = styled.div`
  font-size: 12px;
  color: gray;
  text-align: right;
  margin-bottom: 5px;
`;